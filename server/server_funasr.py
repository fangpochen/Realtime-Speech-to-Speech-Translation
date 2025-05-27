""" Server for real-time translation and voice synthesization using FunASR """
from typing import Dict
from queue import Queue
import select
import socket
import pyaudio
import torch
from models.speech_recognition_funasr import FunASRSpeechRecognitionModel
from models.text_to_speech import TextToSpeechModel
from models.translator import Translator

class AudioSocketServerFunASR:
    """ Class that handles real-time translation and voice synthesization using FunASR
        Socket input -> FunASR -> text -> TextToSpeech -> Socket output
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    PORT = 4444
    # Number of unaccepted connections before server refuses new connections.
    BACKLOG = 5
    
    def __init__(self, funasr_model="paraformer-zh"):
        self.audio = pyaudio.PyAudio()
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Let kernel know we want to reuse the same port for restarting the server
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TODO: For multiple concurrent users we will need more queues
        self.data_queue : Queue = Queue()

        # Initialize the FunASR transcriber model
        self.transcriber = FunASRSpeechRecognitionModel(
            model_name=funasr_model,
            data_queue=self.data_queue,
            generation_callback=self.handle_generation,
            final_callback=self.handle_transcription
        )
        
        self.text_to_speech = TextToSpeechModel(callback_function=self.handle_synthesize)
        self.text_to_speech.load_speaker_embeddings()
        
        # 初始化翻译器
        self.translator = Translator(service="google")  # 使用Google翻译
        
        self.read_list = []

    def __del__(self):
        self.audio.terminate()
        self.transcriber.stop()
        self.serversocket.shutdown()
        self.serversocket.close()
        
    def handle_generation(self, packet: Dict):
        """ Placeholder function for transcription"""
        pass
        
    def handle_transcription(self, packet: str, client_socket):
        """ Callback function to put finalized transcriptions into TTS"""
        print(f"识别结果: {packet}")
        
        # 翻译为英文
        translated_text = self.translator.translate_to_english(packet)
        print(f"翻译结果: {translated_text}")
        
        # 合成英文语音
        self.text_to_speech.synthesise(translated_text, client_socket)
        
    def handle_synthesize(self, audio: torch.Tensor, client_socket):
        """ Callback function to stream audio back to the client"""
        self.stream_numpy_array_audio(audio, client_socket)

    def start(self):
        """ Starts the server"""
        self.transcriber.start(16000, 2)
        print(f"FunASR Server listening on port {self.PORT}")
        self.serversocket.bind(('', self.PORT))
        self.serversocket.listen(self.BACKLOG)
        # Contains all of the socket connections
        self.read_list = [self.serversocket]

        try:
            while True:
                readable, _, _ = select.select(self.read_list, [], [])
                for s in readable:
                    if s is self.serversocket:
                        (clientsocket, address) = self.serversocket.accept()
                        self.read_list.append(clientsocket)
                        print("Connection from", address)
                    else:
                        try:
                            data = s.recv(4096)
                            if data:
                                self.data_queue.put((s, data))
                            else:
                                self.read_list.remove(s)
                                print("Disconnection from", address)
                        except ConnectionResetError:
                            self.read_list.remove(s)
                            print("Client crashed from", address)
        except KeyboardInterrupt:
            pass
        print("Performing server cleanup")
        self.audio.terminate()
        self.transcriber.stop()
        self.serversocket.shutdown(socket.SHUT_RDWR)
        self.serversocket.close()
        print("Sockets cleaned up")
        
    def stream_numpy_array_audio(self, audio, client_socket):
        """ Streams audio back to the client"""
        try:
            if client_socket and hasattr(client_socket, 'sendall'):
                audio_bytes = audio.numpy().tobytes()
                client_socket.sendall(audio_bytes)
                print(f"✅ 音频已发送到客户端，大小: {len(audio_bytes)} bytes")
            else:
                print("⚠️  客户端连接已断开，无法发送音频")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"❌ 发送音频失败: {e}")
            if client_socket in self.read_list:
                self.read_list.remove(client_socket)

if __name__ == "__main__":
    server = AudioSocketServerFunASR(funasr_model="paraformer-zh")
    server.start() 