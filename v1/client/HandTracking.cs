using System;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

using Microsoft;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.Input;

using System.Net;
using System.Net.Sockets;
using System.Threading;

public class HandTracking : MonoBehaviour
{
    public const int jointsNum = 26;

    public GameObject jointObject;
    public Vector3 scale = new Vector3(0.01f, 0.01f, 0.01f);
    public Color[] jointsColor = new Color[26] {
        Color.yellow, Color.gray, Color.magenta, Color.magenta, Color.magenta,
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
    public float tmpDistL = 1f;
    public float tmpDistR = 1f;

    Vector3[] jointsL = new Vector3[26];
    Vector3[] jointsR = new Vector3[26];
    static Vector3[] jointsR_Receive = new Vector3[26];

    List<GameObject> jointObjectsL = new List<GameObject>();
    List<GameObject> jointObjectsR = new List<GameObject>();

    List<LineRenderer> fingerBonesL = new List<LineRenderer>();
    List<LineRenderer> fingerBonesR = new List<LineRenderer>();

    List<LineRenderer> lrWords = new List<LineRenderer>();
    int count = 0;
    MixedRealityPose pose;
    

    // Start is called before the first frame update
    void Start()
    {
        Connect(severIp, severPort);
        Receive(m_socket);
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

    // Update is called once per frame
    void Update()
    {
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
            }

            if (HandJointUtils.TryGetJointPose((TrackedHandJoint)(i + 1), Handedness.Right, out pose))
            {
                jointsR[i] = pose.Position;
                jointObjectsR[i].GetComponent<Renderer>().enabled = true;
            }

        }
        if (count++ % 4 == 0)
            Send(m_socket, jointsR);
            

        // set joints position
        for (int i = 0; i < jointsNum; i++)
        {
            jointObjectsL[i].transform.position = jointsL[i];
            jointObjectsL[i].GetComponent<Renderer>().material.color = jointsColor[i];
            jointObjectsL[i].transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            jointObjectsR[i].transform.position = jointsR_Receive[i];
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
      
        
        // judge wirting characters
        DrawLine(jointsL[5], jointsL[10], true);
        DrawLine(jointsR_Receive[5], jointsR_Receive[10], false);

    }
    


    void DrawLine(Vector3 indexTip, Vector3 thumbTip, bool isLeft)
    {
        float distIndexTip2ThumbTip = Vector3.Distance(indexTip, thumbTip);

        if (distIndexTip2ThumbTip < minDist && distIndexTip2ThumbTip != 0)
        {
            // go to draw a word
            if ((tmpDistL > minDist && isLeft) || (tmpDistR > minDist && !isLeft)) // establish a LineRenderer
            {
                Color randomLineColor = new Color(
                    UnityEngine.Random.Range(0f, 1f),
                    UnityEngine.Random.Range(0f, 1f),
                    UnityEngine.Random.Range(0f, 1f)
                );
                LineRenderer lr = InitialLine(randomLineColor, randomLineColor, lineWidth, lineWidth, 0);
                lrWords.Add(lr);
                //Debug.Log(distIndexTip2ThumbTip + "-------CREATE LINE #" + lrWords.Count + "--------" + isLeft);
            }
            else
            {
                LineRenderer lr = lrWords[lrWords.Count - 1];
                lr.positionCount++;
                lr.SetPosition(lr.positionCount - 1, indexTip);

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


    #region TcpClient
    public static string severIp = "219.224.168.98";
    public static int severPort = 5000;
    public static Socket m_socket;
    public static int totalBytesReceived = 0;
    public const int dataLength = jointsNum*12;

    private static ManualResetEvent connectDone = new ManualResetEvent(false);
    private static ManualResetEvent sendDone = new ManualResetEvent(false);
    private static ManualResetEvent receiveDone = new ManualResetEvent(false);

    #region Connect
    public void Connect(string ip, int port)
    {
        try
        {
            Debug.Log("connect begin");

            m_socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            IPEndPoint remoteEP = new IPEndPoint(IPAddress.Parse(ip), port);
            
            m_socket.BeginConnect(remoteEP, new AsyncCallback(ConnectCallback), m_socket);
            connectDone.WaitOne(2000);
        }
        catch (SocketException e)
        {
            Console.WriteLine(e.ToString());
            Debug.Log("connect failed");
        }
    }

    private void ConnectCallback(IAsyncResult ar)
    {
        try
        {
            Socket client = (Socket)ar.AsyncState;
            client.EndConnect(ar);
            //Console.WriteLine("Socket connect to {0}", client.RemoteEndPoint.ToString());
            connectDone.Set();
            Debug.Log("connect to " + client.RemoteEndPoint.ToString()+ " success");
        }
        catch (SocketException e)
        {
            Debug.Log("connect failed");
            Console.WriteLine(e.ToString());
        }
    }
    #endregion

    #region Receive
    public class FrameBuffer
    {
        public Socket workSocket = null;

        public const int BufferSize = dataLength;

        public byte[] sourceBuffer = new byte[BufferSize];
        public byte[] targetBuffer = new byte[BufferSize];

        public int cursor = 0;
        public int frameCount = 0;

        public bool Copy(int bytesRead)
        {
            if (cursor + bytesRead < BufferSize)
            {
                System.Buffer.BlockCopy(sourceBuffer, 0, targetBuffer, cursor, bytesRead);
                cursor = (cursor + bytesRead) % BufferSize;
                return false;
            }
            else
            {
                ++frameCount;
                Debug.Log("proccessing No." + frameCount + "frame");

                System.Buffer.BlockCopy(sourceBuffer, 0, targetBuffer, cursor, BufferSize - cursor);
                BytesToVector3(targetBuffer);
                System.Buffer.BlockCopy(sourceBuffer, BufferSize - cursor, targetBuffer, 0, cursor + bytesRead - BufferSize);

                cursor = (cursor + bytesRead) % BufferSize;
                return true;
            }
        }

        public void BytesToVector3(byte[] ReceiveBytes)
        {
            int datalength = (int)(ReceiveBytes.Length / 12);
            byte[] TempBytes = new byte[4];

            for (int i = 0; i < datalength; i++)
            {
                Buffer.BlockCopy(ReceiveBytes, i * 12 + 0, TempBytes, 0, 4);
                float x = BitConverter.ToSingle(TempBytes, 0);
                Buffer.BlockCopy(ReceiveBytes, i * 12 + 4, TempBytes, 0, 4);
                float y = BitConverter.ToSingle(TempBytes, 0);
                Buffer.BlockCopy(ReceiveBytes, i * 12 + 8, TempBytes, 0, 4);
                float z = BitConverter.ToSingle(TempBytes, 0);

                jointsR_Receive[i] = new Vector3(x, y, z);
            }
        }
    }
    private void Receive(Socket client)
    {
        try
        {
            if (client == null || !client.Connected) return;
            FrameBuffer frame = new FrameBuffer();
            frame.workSocket = client;
            client.BeginReceive(frame.sourceBuffer, 0, FrameBuffer.BufferSize, 0, new AsyncCallback(ReceiveCallback), frame);
        }
        catch (SocketException e)
        {
            Console.WriteLine(e.ToString());
        }
    }

    private void ReceiveCallback(IAsyncResult ar)
    {
        try
        {
            FrameBuffer frame = (FrameBuffer)ar.AsyncState;
            Socket client = frame.workSocket;
            int bytesRead = client.EndReceive(ar);

            if (bytesRead > 0)
            {
                totalBytesReceived += bytesRead;

                bool frameUpdated = frame.Copy(bytesRead);
                if (frameUpdated)
                {
                    //一帧数据接收完成，进行后续处理
                    
                }

                client.BeginReceive(frame.sourceBuffer, 0, FrameBuffer.BufferSize, 0, new AsyncCallback(ReceiveCallback), frame);
            }
            else
            {
                Connect(severIp, severPort);
            }
        }
        catch (SocketException e)
        {
            Debug.Log("receive failed");
            Console.WriteLine(e.ToString());
        }
    }
    #endregion

    #region Send
    private void Send(Socket client, Vector3[] data)
    {
        if (client == null || !client.Connected) return;
        byte[] byteData = Vector3ToBytes(data);
        //Debug.Log("send  " + " " + data[0].x );
        client.BeginSend(byteData, 0, byteData.Length, 0, new AsyncCallback(SendCallback), client);
    }

    private void SendCallback(IAsyncResult ar)
    {
        try
        {
            Socket client = (Socket)ar.AsyncState;
            int bytesSent = client.EndSend(ar);
            //Console.WriteLine("Sent {0} bytes to sever", bytesSent);
            //Debug.Log("send "+ bytesSent +" to sever" );
        }
        catch (SocketException e)
        {
            Console.WriteLine(e.ToString());
        }
    }
   
    public static byte[] Vector3ToBytes(Vector3[] SendData)
    {
        int datalength = SendData.Length;
        byte[] SendBytes = new byte[datalength*12];

        for (int i = 0; i < datalength; i++)
        {
            BitConverter.GetBytes(SendData[i].x).CopyTo(SendBytes, i * 12 + 0);

            BitConverter.GetBytes(SendData[i].y).CopyTo(SendBytes, i * 12 + 4);

            BitConverter.GetBytes(SendData[i].z).CopyTo(SendBytes, i * 12 + 8);
        }
        return SendBytes;
    }
    #endregion

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
