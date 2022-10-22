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
using System.Threading.Tasks;

public class HandTrackingApp : MonoBehaviour
{
    public const string serverIp = "219.224.168.98";
    public const int serverPort = 5000;
    
    public TcpConnect tcp = new TcpConnect();

    public void Start()
    {
        // Connect the server...
        tcp.Connect(serverIp, serverPort);
    }

    public void Update()
    {
        // Re-Connect the server...
        if (tcp.connectStatus == 0 || tcp.connectStatus == -1) tcp.Connect(serverIp, serverPort);
    }

    #region MRTK hand tracing
    public class HandTracking
    {

    }
    #endregion
#


    #region tcp connect
    public class TcpConnect
    {
        public Socket m_socket;
        // 0: no connecting (default) 1: is connecting
        // 2: connect successfully -1: connect faild
        public int connectStatus = 0;

        public static ManualResetEvent connectDone = new ManualResetEvent(false);

        // Connect
        public void Connect(string ip, int port)
        {
            try
            {
                connectStatus = 1;
                Debug.Log("Connecting...");

                m_socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                IPEndPoint remoteEP = new IPEndPoint(IPAddress.Parse(ip), port);

                m_socket.BeginConnect(remoteEP, new AsyncCallback(ConnectCallback), m_socket);
                connectDone.WaitOne(200);
            }
            catch (SocketException e)
            {
                connectStatus = -1;
                Debug.Log("Connect failed..." + e.ToString());
            }
        }

        public void ConnectCallback(IAsyncResult ar)
        {
            try
            {
                Socket client = (Socket)ar.AsyncState;
                client.EndConnect(ar);

                connectDone.Set();

                connectStatus = 2;
                Debug.Log("================CONNECT SUCCESSFULLY================");
            }
            catch (SocketException e)
            {
                connectStatus = -1;
                Debug.Log("Connect failed..." + e.ToString());
            }
        }
    }



    
#endregion
}
