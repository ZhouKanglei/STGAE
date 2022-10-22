using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;


public class TcpClient : MonoBehaviour
{
    public static string severIp = "192.168.1.105";
    public static int severPort = 7777;
    public Socket m_socket;

    public const int dataLength = 524288;
    public static byte[] receiveBuffer = new byte[dataLength]; 
    public static Int32[] frameData = new Int32[dataLength];
    public static int totalBytesReceived = 0;

    private static ManualResetEvent connectDone = new ManualResetEvent(false);
    private static ManualResetEvent sendDone = new ManualResetEvent(false);
    private static ManualResetEvent receiveDone = new ManualResetEvent(false);
    private static ManualResetEvent computebufferSetDone = new ManualResetEvent(false);

    public ComputeBuffer shaderData;
    bool SetDataStart = false;
    bool timerStart = false;
    System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();

    public void Start() {
        
        shaderData = new ComputeBuffer(dataLength, sizeof(Int32));
        shaderData.SetData(frameData);
        GetComponent<Renderer>().material.SetBuffer("_DenLight", shaderData);
        Connect(severIp, severPort);
        Receive(m_socket);
    }

    public void Update(){
        if (SetDataStart)
        {
            //Debug.Log(" set begin");
            shaderData.SetData(frameData);
            //Debug.Log(" set success");

            /*int[] temp = new int[dataLength];
            shaderData.GetData(temp);
            Array.Sort(temp,262144,262144);
            Debug.Log(" maxvalue = " + temp[frameData.Length-1]);*/

            SetDataStart = false;
            //computebufferSetDone.Set();
            
        }
        //GetComponent<Renderer>().material.SetBuffer("_DenLight", shaderData);

    }

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
            //Debug.Log("connect success");
        }
        catch (SocketException e)
        {
            Debug.Log("connect failed");
            Console.WriteLine(e.ToString());
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
            Debug.Log("connect success");
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

        public int cursor = 0;
        public int frameCount = 0;

        public byte[] targetBuffer = new byte[BufferSize];

        public bool Copy(int bytesRead)
        {           
            if(cursor + bytesRead < BufferSize)
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
                ByteToData(targetBuffer);
                System.Buffer.BlockCopy(sourceBuffer, BufferSize - cursor, targetBuffer, 0, cursor + bytesRead - BufferSize);
                cursor = (cursor + bytesRead) % BufferSize;
                return true;
            }
        }
        public void ByteToData(byte[] inputData)
        {
            byte[] temp = new byte[4 * inputData.Length];
            Parallel.For(0, inputData.Length, i =>
            {
                System.Buffer.BlockCopy(inputData, i, temp, i * 4, 1);
                frameData[i] = BitConverter.ToInt32(temp, i * 4);
            });
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
            if (timerStart == false)
            {
                Debug.Log("receive begin");
                watch.Start();
                timerStart = true;
            }

            FrameBuffer frame = (FrameBuffer)ar.AsyncState;
            Socket client = frame.workSocket;
            int bytesRead = client.EndReceive(ar);

            if (bytesRead > 0)
            {
                
                totalBytesReceived += bytesRead;

                bool frameUpdated = frame.Copy(bytesRead);
                if (frameUpdated)
                {
                    //if(shaderData.IsValid()) Debug.Log("shaderData invalid");
                    //int[] temp = frameData;
                    //Array.Sort(temp,262144,262144);
                    //Debug.Log(frame.frameCount + " maxvalue = " + temp[frameData.Length-1]);
                    /*Debug.Log("test begin");
                    ComputeBufferSetData(shaderData, frameData);
                    Debug.Log("test success");*/
                    SetDataStart = true;

                    //computebufferSetDone.Reset();
                    //computebufferSetDone.WaitOne();
                    //GetComponent<Renderer>().material.SetBuffer("_DenLight", shaderData);
                }

                client.BeginReceive(frame.sourceBuffer, 0, FrameBuffer.BufferSize, 0, new AsyncCallback(ReceiveCallback), frame);

                //Debug.Log("receive " + bytesRead + "bytes from sever "   + frameData[32767]);

                //Debug.Log("receive " + bytesRead + " bytes");
            }
            else
            {
                //Connect(severIp, severPort);
                client.Close();
                Debug.Log("receive end,total " + totalBytesReceived + " bytes received and " + frame.frameCount + " processed");
                watch.Stop();
                Debug.Log("耗时 ：" + watch.ElapsedMilliseconds);
            }
        }
        catch (SocketException e)
        {
            Debug.Log("receive failed");
            Console.WriteLine(e.ToString());
        }
    }
    private void ComputeBufferSetData(ComputeBuffer buffer, int[] data)
    {
        buffer.SetData(data);
    }
    #endregion

    #region Send
    private void Send(Socket client, Vector3[] data)
    {
        try
        {
            if (client == null || !client.Connected) return;
            byte[] byteData = new byte[4];
            //Debug.Log("send  " + " " + data[0].x );
            client.BeginSend(byteData, 0, byteData.Length, 0, new AsyncCallback(SendCallback), client);
        }
        catch (SocketException e)
        {
            Debug.Log("send failed");
            Console.WriteLine(e.ToString());
        }
        
    }

    private void SendCallback(IAsyncResult ar)
    {
        try
        {
            Socket client = (Socket)ar.AsyncState;
            int bytesSent = client.EndSend(ar);
            Console.WriteLine("Sent {0} bytes to sever", bytesSent);
            //Debug.Log("send "+ bytesSent +" to sever" );
            sendDone.Set();
        }
        catch (SocketException e)
        {
            Debug.Log("send failed");
            Console.WriteLine(e.ToString());
        }
    }
    #endregion
}
