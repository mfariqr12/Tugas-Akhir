import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import librosa
import numpy as np
import pickle
from threading import Thread
import time

def getMFCC(signal, sample_rate):
    fixed_sample = 22050 * 3
    
    if len(signal) >= fixed_sample:
        signal = signal[:fixed_sample]
    
    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=1024, hop_length=512)
    
    return MFCCs.T

class VowelClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("Klasifiasi Suara Sirene")
        master.geometry("650x420")
        master.configure(bg="#4a4a4a")

        self.page1 = Page1(master, self)

    def show_page1(self, option):
        self.page1.show(option)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()
        self.frame.configure(bg="#4a4a4a")

        self.label = tk.Label(self.frame, text="Tekan untuk Merekam", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.record_button = tk.Button(self.frame, text="Rekam Suara", font=("Helvetica", 28), bg="white", fg="black", command=self.record_audio)
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        # self.timeLen = tk.Label(self.frame, text="", font=("Helvetica", 28), bg="#4a4a4a", fg="white")

        self.label.pack(pady=10)
        self.record_button.pack(pady=20)
        self.prediction_label.pack(pady=10)
        # self.timeLen.pack(pady=10)

    def show(self, option):
        self.frame.pack()

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):

        self.label.config(text="Perekaman suara")
        self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 3
            file_name = "sound.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara berhasil")
            self.master.update()

            # start_time = time.time()

            with open("HMModel.pkl", 'rb') as f: 
                hmmModels = pickle.load(f)
            signal, sample_rate = librosa.load(file_name)
            mfccs = getMFCC(signal, sample_rate)

            scoreList = {}
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(mfccs)
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)

            # end_time = time.time()
            # elapsed_time = end_time - start_time

            self.prediction_label.config(text=f"Prediksi: {predict}")
            # self.timeLen.config(text=f"Waktu: {elapsed_time:.2f}s")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            print(f"Exception: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VowelClassificationApp(root)
    root.mainloop()