using System;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.UI;

using Microsoft;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.Input;


using System.Net;
using System.Net.Sockets;
using System.Threading;
public class HandTracking : MonoBehaviour
{
    #region hand variables
    public int jointsNum = 26;
    public GameObject jointObject;
    public Vector3 scale = new Vector3(0.01f, 0.01f, 0.01f);
    public Color[] jointsColor = new Color[26] {
        Color.yellow, Color.clear, Color.magenta, Color.magenta, Color.magenta,
        Color.magenta, Color.cyan, Color.cyan, Color.cyan, Color.cyan,
        Color.cyan, Color.red, Color.red, Color.red, Color.red,
        Color.red, Color.green, Color.green, Color.green, Color.green,
        Color.green, Color.blue, Color.blue, Color.blue, Color.blue, Color.blue
    };

    public GameObject line;
    public float lineWidth = 0.005f;
    public int fingersNum = 5;
    public int[,] fingers = new int[5, 6] {
        {0, 2, 3, 4, 5, 5},
        {0, 6, 7, 8, 9, 10},
        {0, 11, 12, 13, 14, 15},
        {0, 16, 17, 18, 19, 20},
        {0, 21, 22, 23, 24, 25}
    };
    public Color[] fingersColor = new Color[5] {
        Color.magenta, Color.cyan, Color.red, Color.green, Color.blue
    };

    public float minDist = 0.02f;
    public float minDistRecieve = 0.15f;
    public float tmpDistL = 1f;
    public float tmpDistR = 1f;
    public float tmpDistLRecieve = 1f;
    public float tmpDistRRecieve = 1f;

    Vector3[] jointsL = new Vector3[26];
    Vector3[] jointsR = new Vector3[26];
    Vector3[] jointsSend = new Vector3[26];
    Vector3[] jointsRecieve = new Vector3[26];

    List<GameObject> jointObjectsL = new List<GameObject>();
    List<GameObject> jointObjectsR = new List<GameObject>();

    List<LineRenderer> fingerBonesL = new List<LineRenderer>();
    List<LineRenderer> fingerBonesR = new List<LineRenderer>();

    // two dynamic line render container, one for the orignal data, the other for the recieving data
    List<LineRenderer> lrWords = new List<LineRenderer>();
    List<LineRenderer> lrWordsRecieve = new List<LineRenderer>();

    // Log the line event: 0 is not in the line, 1 is in line, 2 is the starting point of the line
    //                     3 is the end of the line, 4 is cleaning the forward line
    List<int> lineEvent = new List<int>();
    List<Vector3> pointGroundTruth = new List<Vector3>();
    List<Vector3> point = new List<Vector3>();
    int lastVisitedPoint = 17;

    MixedRealityPose pose;
    #endregion

    #region tcp variables
    public string serverIp = "219.224.168.98";
    public int serverPort = 5000;
    public Socket socket;

    public int count = 0;
    public int sendCount = 0;
    public int recieveCount = 0;
    public int recieveLastCount = 0;
    public int temporalSize = 36;
    public int sendFrequency = 1;
    public byte[] receiveBuffer = new byte[26 * 3 * 4];
    
    public bool isHandInScene = false;
    public bool connectStatus = false;
    public bool connectSuccess = false;
    public bool leftHandInScene = false;
    public bool rightHandInScene = false;

    private static ManualResetEvent connectDone = new ManualResetEvent(false);
    private static ManualResetEvent sendDone = new ManualResetEvent(false);
    private static ManualResetEvent receiveDone = new ManualResetEvent(false);
    #endregion


    #region UI
    public Text connectionShow;
    public Text handStatusShow;
    public Text countShow;
    public Text timeShow;
    public Text recieveShow;
    public Text sendShow;

    int hour;
    int minute;
    int second;
    int milliSecond;

    float timeSpend = 0.0f;
    float recieveTimeSpend = 0.0f; // the denominator cannot be zero
    float sendTimeSpend = 0.0f; // the denominator cannot be zero
    float recieveTimeSpendStart = 0.0001f; // the denominator cannot be zero
    float sendTimeSpendStart = 0.0001f; // the denominator cannot be zero

    bool isSend = false;
    bool isRecieve = false;
    #endregion

    // Start is called before the first frame update
    void Start()
    {
        // initialize the joint and bone object
        BoneJointInit();

        // connect the server
#if UNITY_EDITOR
        Debug.Log("Connecting...");
#endif
        Connect(serverIp, serverPort);
    }

    // Update is called once per frame
    void Update()
    {
        // variable initialization
        variableInit();

        // render bone and joints
        BoneJointRender();

        // recieve and send data
        count += 1; // current updating frame
        if (connectStatus == true) // judge whether connection is normal
        {   
            if (count % sendFrequency == 0 && isHandInScene) // hand in scene && send every #sendFrequency frames
            {
                jointsSend = rightHandInScene ? jointsR : jointsL;

                if (sendCount == 0) sendTimeSpendStart = timeSpend;
                Send(socket, jointsSend); // send right hand joints to server
                if (sendCount >= temporalSize)
                {
                    if (recieveCount == 0) recieveTimeSpendStart = timeSpend;
                    Receive(socket, receiveBuffer); // recieve data from server

                    // judge whether wirting characters
                    if (rightHandInScene) DrawLine(jointsR[5], jointsR[10], false, true); 
                    else DrawLine(jointsL[5], jointsL[10], true, true);

                    // judge whether clearing all paintings
                    if (rightHandInScene) IsClear(jointsR[5], jointsR[15]);
                    else IsClear(jointsL[5], jointsL[15]);

                    //Debug.Log(lineEvent.Count + " Event " + lineEvent[lineEvent.Count - 1]);
                }
            }  
        }
        else if (connectStatus == false)
        {
#if UNITY_EDITOR
            Debug.Log("Re-connecting...");
#endif
            Connect(serverIp, serverPort); // re-connecting...
        }

        if (recieveCount <= lineEvent.Count && recieveCount > 18 && lastVisitedPoint < recieveCount - 18)
        {
            //DrawLineRecieve(point[lastVisitedPoint+1], point[lastVisitedPoint+1], false, false);
            lastVisitedPoint += 1;
        }
    }

    #region render bone and joint
    void variableInit()
    {
        #region UI text to show
        //connectionShow = GameObject.Find("Canvas/Main/ConnectionShow").GetComponent<Text>();
        connectionShow.text = "Connect the remote server " 
            + serverIp + "/" + serverPort + "  Status: " + connectSuccess;
        //handStatusShow = GameObject.Find("Canvas/Main/HandStatusShow").GetComponent<Text>();
        if (recieveCount >= 1)
            handStatusShow.text = "Hand in the scene: " + isHandInScene + "  Left: " 
                + leftHandInScene + "  Right: " + rightHandInScene
                + "  Event: " + lineEvent[recieveCount - 1];
        else
            handStatusShow.text = "Hand in the scene: " + isHandInScene + "  Left: "
                + leftHandInScene + "  Right: " + rightHandInScene
                + "  Event: None";
        
        //countShow = GameObject.Find("Canvas/Main/CountShow").GetComponent<Text>();

        timeSpend += Time.deltaTime;

        hour = (int)timeSpend / 3600;
        minute = ((int)timeSpend - hour * 3600) / 60;
        second = (int)timeSpend - hour * 3600 - minute * 60;
        milliSecond = (int)((timeSpend - (int)timeSpend) * 1000);

        //timeShow = GameObject.Find("Canvas/Main/TimeShow").GetComponent<Text>();
        timeShow.text = System.DateTime.Now.ToString() + "  "
            + string.Format("{0:D2}:{1:D2}:{2:D2}.{3:D3}", hour, minute, second, milliSecond);

        //recieveShow = GameObject.Find("Canvas/Main/RecieveShow").GetComponent<Text>();
        recieveShow.text = "Thumb:" + jointsRecieve[5].ToString("f4") + "  Index: " + jointsRecieve[10].ToString("f4");

        //sendShow = GameObject.Find("Canvas/Main/SendShow").GetComponent<Text>();
        sendShow.text = "Thumb:" + jointsSend[5].ToString("f4") + "  Index: " + jointsSend[10].ToString("f4");

        if (isSend) sendTimeSpend = timeSpend;
        if (isRecieve) recieveTimeSpend = timeSpend;

        countShow.text = "Send: " + sendCount + " [" + (sendCount / (sendTimeSpend - sendTimeSpendStart)).ToString("f1") + "fps]"
                + "  Recieve: " + recieveCount + " [" + (recieveCount / (recieveTimeSpend - recieveTimeSpendStart)).ToString("f1") + "fps]"
                + "  Current: " + (count / 1000).ToString("f0")  + " [" + (count / timeSpend).ToString("f1") + "fps]";
        #endregion


        jointsL = new Vector3[26];
        jointsR = new Vector3[26];
        isRecieve = false;
        isSend = false;
    }

    void BoneJointInit()
    {
        // initialization
        for (int i = 0; i < jointsNum; i++)
        {
            GameObject obj1 = Instantiate(jointObject, this.transform);
            jointObjectsL.Add(obj1);

            GameObject obj2 = Instantiate(jointObject, this.transform);
            jointObjectsR.Add(obj2);
        }

        for (int i = 0; i < fingersNum; i++)
        {
            LineRenderer lr1 = InitialLine(fingersColor[i], fingersColor[i], lineWidth, lineWidth, 6);
            fingerBonesL.Add(lr1);

            LineRenderer lr2 = InitialLine(fingersColor[i], fingersColor[i], lineWidth, lineWidth, 6);
            fingerBonesR.Add(lr2);
        }
    }

    void BoneJointRender()
    {
        // initilize variables...
        variableInit();

        // only render if hand is tracked
        for (int i = 0; i < jointsNum; i++)
        {
            jointObjectsL[i].GetComponent<Renderer>().enabled = false;
            jointObjectsR[i].GetComponent<Renderer>().enabled = false;
        }

        for (int i = 0; i < fingersNum; i++)
        {
            fingerBonesL[i].enabled = false;
            fingerBonesR[i].enabled = false;
        }

        // obtain all the joints of left and right hands respectively
        for (int i = 0; i < jointsNum; i++)
        {
            if (HandJointUtils.TryGetJointPose((TrackedHandJoint)(i + 1), Handedness.Left, out pose))
            {
                jointsL[i] = pose.Position;
                jointObjectsL[i].GetComponent<Renderer>().enabled = true;

                //Debug.Log("Left hand, joint - " + (i + 1) + " "
                //    + (TrackedHandJoint)(i + 1)
                //    + ": " + jointsL[i]);
            }

            if (HandJointUtils.TryGetJointPose((TrackedHandJoint)(i + 1), Handedness.Right, out pose))
            {
                jointsR[i] = pose.Position;
                jointObjectsR[i].GetComponent<Renderer>().enabled = true;

                //Debug.Log("Right hand, joint - " + (i + 1) + " "
                //    + (TrackedHandJoint)(i + 1)
                //    + ": " + jointsR[i]);
            }

        }

        // set joints position
        for (int i = 0; i < jointsNum; i++)
        {
            jointObjectsL[i].transform.position = jointsL[i];
            jointObjectsL[i].GetComponent<Renderer>().material.color = jointsColor[i];
            jointObjectsL[i].transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            jointObjectsR[i].transform.position = jointsR[i];
            jointObjectsR[i].GetComponent<Renderer>().material.color = jointsColor[i];
            jointObjectsR[i].transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        }

        // draw bone line
        for (int i = 0; i < fingersNum; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                if (jointObjectsL[0].GetComponent<Renderer>().enabled)
                {
                    fingerBonesL[i].enabled = true;
                    fingerBonesL[i].SetPosition(j, jointsL[fingers[i, j]]);
                }

                if (jointObjectsR[0].GetComponent<Renderer>().enabled)
                {
                    fingerBonesR[i].enabled = true;
                    fingerBonesR[i].SetPosition(j, jointsR[fingers[i, j]]);
                }
            }
        }

        // judge whelter any hand is in scene 
        if (!jointObjectsL[5].GetComponent<Renderer>().enabled && !jointObjectsR[5].GetComponent<Renderer>().enabled)
        {
            isHandInScene = false;
            leftHandInScene = false;
            rightHandInScene = false;
        }
        else
        {
            isHandInScene = true;
            leftHandInScene = jointObjectsL[5].GetComponent<Renderer>().enabled ? true : false;
            rightHandInScene = jointObjectsR[5].GetComponent<Renderer>().enabled ? true : false;
        }

    }   
    void IsClear(Vector3 thumbTip, Vector3 middleTip)
    {
        float distThumbTip2MiddleTip = Vector3.Distance(thumbTip, middleTip);

        if (distThumbTip2MiddleTip < minDist && distThumbTip2MiddleTip != 0)
        {
            lineEvent.Add(4); // clear the line

            foreach (LineRenderer lr in lrWords)
            {
                lr.enabled = false;
            }
#if UNITY_EDITOR 
            Debug.Log(distThumbTip2MiddleTip + "-------CLEAR ALL LINE #" + lrWords.Count + "--------");
#endif
        }
    }

    void DrawLine(Vector3 thumbTip, Vector3 indexTip, bool isLeft, bool isRandomColor)
    {
        float distIndexTip2ThumbTip = Vector3.Distance(indexTip, thumbTip);

        if (distIndexTip2ThumbTip > minDist)
        {
            if ((tmpDistL <= minDist && isLeft) || (tmpDistR <= minDist && !isLeft))
            {
                lineEvent.Add(3); // end the line
            }
            else
            {
                lineEvent.Add(0); // not in any line
            }
        }

        if (distIndexTip2ThumbTip < minDist && distIndexTip2ThumbTip != 0)
        {
            // go to draw a word
            if ((tmpDistL > minDist && isLeft) || (tmpDistR > minDist && !isLeft)) // establish a LineRenderer
            {
                lineEvent.Add(2); // crate a new line

                Color randomLineColor = Color.gray;
                if (isRandomColor)
                {
                    randomLineColor = new Color(
                        UnityEngine.Random.Range(0f, 1f),
                        UnityEngine.Random.Range(0f, 1f),
                        UnityEngine.Random.Range(0f, 1f)
                    );
                }
                
                LineRenderer lr = InitialLine(randomLineColor, randomLineColor, lineWidth * 1.2f, lineWidth * 1.2f, 0);
                lrWords.Add(lr);
                //Debug.Log(distIndexTip2ThumbTip + "-------CREATE LINE #" + lrWords.Count + "--------" + isLeft);
            }
            else
            {
                lineEvent.Add(1); // point in the line

                LineRenderer lr = lrWords[lrWords.Count - 1];
                lr.positionCount++;
                lr.SetPosition(lr.positionCount - 1, 0.5f * (indexTip + thumbTip));

                //Debug.Log(lr.positionCount + "-------DRAW LINE #" + lrWords.Count + " POINT #" + lrWords.Count + "--------");
            }  
        }

        if (isLeft)
        {
            tmpDistL = distIndexTip2ThumbTip;
        }
        else
        {
            tmpDistR = distIndexTip2ThumbTip;
        }
    }

    void DrawLineRecieve(Vector3 thumbTip, Vector3 indexTip, bool isLeft, bool isRandomColor)
    {
        int e = lineEvent[recieveCount - 1];
        //Debug.Log("Event " + e.ToString() + " Point #" + (lastVisitedPoint + 1) 
        //    + " " + point[lastVisitedPoint + 1] + " Line count: " + lrWordsRecieve.Count);

        switch (e)
        {
            case 0: // the point is not in any line
                { }
                break;
            case 1: // the point is in the line
                {
                    LineRenderer lr = lrWordsRecieve[lrWordsRecieve.Count - 1];
                    lr.positionCount++;
                    lr.SetPosition(lr.positionCount - 1, 0.5f * (indexTip + thumbTip));
                }
                break;
            case 2: // create a new line
                {
                    Color randomLineColor = Color.gray;
                    Debug.Log(isRandomColor + "-------------");
                    if (isRandomColor)
                    {
                        Debug.Log(isRandomColor + "-------------");
                        randomLineColor = new Color(
                            UnityEngine.Random.Range(0f, 1f),
                            UnityEngine.Random.Range(0f, 1f),
                            UnityEngine.Random.Range(0f, 1f)
                        );
                        Debug.Log(isRandomColor + "-------------");
                    }
                    Debug.Log("CREATE LINE-----2------");
                    LineRenderer lr = InitialLine(randomLineColor, randomLineColor, lineWidth * 1.2f, lineWidth * 1.2f, 0);
                    lrWordsRecieve.Add(lr);
                    Debug.Log("CREATE LINE-----3------");
                }
                break;
            case 3:
                {

                    LineRenderer lr = lrWordsRecieve[lrWordsRecieve.Count - 1];
                    lr.positionCount++;
                    lr.SetPosition(lr.positionCount - 1, 0.5f * (indexTip + thumbTip));
                }
                break;
            case 4: // clean the line
                {
                    foreach (LineRenderer lr in lrWordsRecieve)
                    {
                        lr.enabled = false;
                    }
                }
                break;   
        }
    }

    LineRenderer InitialLine(Color startColor, Color endColor, float startWidth, float endWidth, int positionCount)
    {
        GameObject obj = Instantiate(line, this.transform);
        LineRenderer lr = obj.GetComponent<LineRenderer>();
        lr.positionCount = positionCount;
        lr.startColor = startColor;
        lr.endColor = endColor;
        lr.startWidth = startWidth;
        lr.endWidth = endWidth;

        return lr;
    }
    #endregion

    #region tcp client
    void connectInit()
    {
        timeSpend = 0.0f;
        recieveTimeSpend = 0.0f; // the denominator cannot be zero
        sendTimeSpend = 0.0f; // the denominator cannot be zero
        recieveTimeSpendStart = 0.0001f; // the denominator cannot be zero
        sendTimeSpendStart = 0.0001f; // the denominator cannot be zero

        sendCount = 0;
        recieveCount = 0;
        recieveLastCount = -1;

        connectStatus = true;
        connectSuccess = false;
    }


    public void Connect(string ip, int port)
    {
        try
        {
            connectInit();

            connectStatus = true;
            connectSuccess = false;

            socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            IPEndPoint remoteEP = new IPEndPoint(IPAddress.Parse(ip), port);

            //connectDone.Reset();
            socket.BeginConnect(remoteEP, new AsyncCallback(ConnectCallback), socket);
            //connectDone.WaitOne();
        }
        catch (SocketException e)
        {
#if UNITY_EDITOR
            Debug.Log("Connect failed..." + e.ToString());
#endif
        }
    }

    private void ConnectCallback(IAsyncResult ar)
    {
        try
        {
            //connectDone.Set();
            Socket client = (Socket)ar.AsyncState;
            client.EndConnect(ar);

            connectSuccess = true;

#if UNITY_EDITOR
            Debug.Log("======================CONNECT "
                + client.RemoteEndPoint
                + " SUCCESSFULY======================");
#endif

        }
        catch (SocketException e)
        {
            connectStatus = false;

#if UNITY_EDITOR
            Debug.Log("ConnectCallback [connect interupt & Re-connecting...]" + e.ToString());
#endif

        }
    }

    private void Receive(Socket client, byte[] buffer)
    {
        try
        {
            if (client == null || !client.Connected) return;
            //receiveDone.Reset();
            client.BeginReceive(buffer, 0, buffer.Length, 0, new AsyncCallback(ReceiveCallback), client);
            //receiveDone.WaitOne();
        }
        catch (SocketException e)
        {
#if UNITY_EDITOR
            Debug.Log("Recieve: " + e.ToString());
#endif
        }
    }

    private void ReceiveCallback(IAsyncResult ar)
    {
        try
        {
            receiveDone.Set();

            Socket client = (Socket)ar.AsyncState;
            int bytesRead = client.EndReceive(ar);

            if (bytesRead > 0)
            {
                jointsRecieve = BytesToVector3(receiveBuffer);
                recieveCount += 1;
                point.Add(0.5f * (jointsRecieve[5] + jointsRecieve[10]));
                isRecieve = true;
                DrawLineRecieve(point[point.Count - 1], point[point.Count - 1], false, true);
#if UNITY_EDITOR
                Debug.Log("Receive [" + bytesRead + "] bytes from server...");

                //Debug.Log(jointsRecieve[5].ToString("f4") + " " + jointsRecieve[10].ToString("f4"));
                //Debug.Log(recieveCount + " --------- " + lineEvent.Count);
#endif
            }
            else
            {
                //connectStatus = false;
            }
        }
        catch (SocketException e)
        {
#if UNITY_EDITOR
            Debug.Log("[ReceiveCallback]: " + e.ToString());
#endif
            socket.Close();
            connectStatus = false;
        }
    }

    private void Send(Socket client, Vector3[] data)
    {
        pointGroundTruth.Add(0.5f * (data[5] + data[10]));
        if (client == null || !client.Connected) return;
        byte[] byteData = Vector3ToBytes(data);
        client.BeginSend(byteData, 0, byteData.Length, 0, new AsyncCallback(SendCallback), client);
    }

    private void SendCallback(IAsyncResult ar)
    {
        try
        {
            Socket client = (Socket)ar.AsyncState;
            int bytesSent = client.EndSend(ar);
            //sendDone.Set();
            sendCount += 1; // send counter
            isSend = true;


#if UNITY_EDITOR
            Debug.Log("Send [" + bytesSent + "] to server...");
#endif
        }
        catch (SocketException e)
        {
            Debug.Log("SendCallback [Connection interupt & Re-connecting...]: " + e.ToString());
            socket.Close();
            connectStatus = false;
        }
    }
    public byte[] Vector3ToBytes(Vector3[] SendData)
    {
        int datalength = SendData.Length;
        byte[] SendBytes = new byte[datalength * 12];
        for (int i = 0; i < datalength; i++)
        {
            BitConverter.GetBytes(SendData[i].x).CopyTo(SendBytes, i * 12 + 0);
            BitConverter.GetBytes(SendData[i].y).CopyTo(SendBytes, i * 12 + 4);
            BitConverter.GetBytes(SendData[i].z).CopyTo(SendBytes, i * 12 + 8);
        }
        return SendBytes;
    }

    public Vector3[] BytesToVector3(byte[] ReceiveBytes)
    {
        int datalength = (int)(ReceiveBytes.Length / 12);
        Vector3[] ReceiveData = new Vector3[datalength];
        byte[] TempBytes = new byte[4];
        for (int i = 0; i < datalength; i++)
        {
            Buffer.BlockCopy(ReceiveBytes, i * 12 + 0, TempBytes, 0, 4);
            float x = BitConverter.ToSingle(TempBytes, 0);
            Buffer.BlockCopy(ReceiveBytes, i * 12 + 4, TempBytes, 0, 4);
            float y = BitConverter.ToSingle(TempBytes, 0);
            Buffer.BlockCopy(ReceiveBytes, i * 12 + 8, TempBytes, 0, 4);
            float z = BitConverter.ToSingle(TempBytes, 0);

            ReceiveData[i] = new Vector3(x, y, z);
        }
        return ReceiveData;
    }

    #endregion
}


//IndexDistalJoint 10
//The joint nearest the tip of the index finger.

//IndexKnuckle	8	
//The knuckle joint of the index finger.

//IndexMetacarpal	7	
//The lowest joint of the index finger.

//IndexMiddleJoint	9	
//The middle joint of the index finger.

//IndexTip	11	
//The tip of the index finger.

//MiddleDistalJoint	15	
//The joint nearest the tip of the finger.

//MiddleKnuckle	13	
//The knuckle joint of the middle finger.

//MiddleMetacarpal	12	
//The lowest joint of the middle finger.

//MiddleMiddleJoint	14	
//The middle joint of the middle finger.

//MiddleTip	16	
//The tip of the middle finger.

//None	0	
//Palm	2	
//The palm.

//PinkyDistalJoint	25	
//The joint nearest the tip of the pink finger.

//PinkyKnuckle	23	
//The knuckle joint of the pinky finger.

//PinkyMetacarpal	22	
//The lowest joint of the pinky finger.

//PinkyMiddleJoint	24	
//The middle joint of the pinky finger.

//PinkyTip	26	
//The tip of the pinky.

//RingDistalJoint	20	
//The joint nearest the tip of the ring finger.

//RingKnuckle	18	
//The knuckle of the ring finger.

//RingMetacarpal	17	
//The lowest joint of the ring finger.

//RingMiddleJoint	19	
//The middle joint of the ring finger.

//RingTip	21	
//The tip of the ring finger.

//ThumbDistalJoint	5	
//The thumb's first (furthest) joint.

//ThumbMetacarpalJoint	3	
//The lowest joint in the thumb (down in your palm).

//ThumbProximalJoint 4
//The thumb's second (middle-ish) joint.

//ThumbTip	6	
//The tip of the thumb.

//Wrist	1	
//The wrist.
