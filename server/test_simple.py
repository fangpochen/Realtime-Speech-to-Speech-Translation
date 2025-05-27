#!/usr/bin/env python3
"""简化的测试服务器 - 不使用Whisper模型"""
import socket
import threading
import time

class SimpleTestServer:
    def __init__(self, port=4444):
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def handle_client(self, client_socket, address):
        print(f"客户端连接: {address}")
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                # 简单回声 - 发送接收到的数据回去
                print(f"收到数据长度: {len(data)}")
                client_socket.send(data)
        except Exception as e:
            print(f"客户端错误: {e}")
        finally:
            client_socket.close()
            print(f"客户端断开: {address}")
    
    def start(self):
        self.server_socket.bind(('', self.port))
        self.server_socket.listen(5)
        print(f"测试服务器启动，监听端口 {self.port}")
        
        try:
            while True:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n服务器关闭")
        finally:
            self.server_socket.close()

if __name__ == "__main__":
    server = SimpleTestServer()
    server.start() 