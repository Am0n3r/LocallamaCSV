import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import requests
import threading
import os

DEFAULT_CONFIG = {
    "api_url": "http://localhost:11434/api/generate",
    "model": "gemma3:1b",
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 2048,
    "num_ctx": 2048,
    "repeat_penalty": 1.1,
    "seed": "",
    "timeout": 60,
}

class LLMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CSV LLM Generator")
        self.geometry("600x600")
        ctk.set_appearance_mode("System")

        self.file_path = ""
        self.config = DEFAULT_CONFIG.copy()

        self.model_entry = self.add_labeled_entry("Модель Ollama", self.config["model"])
        self.temperature_entry = self.add_labeled_entry("Температура", str(self.config["temperature"]))
        self.top_p_entry = self.add_labeled_entry("Top-p", str(self.config["top_p"]))
        self.max_tokens_entry = self.add_labeled_entry("Макс. токенов", str(self.config["max_tokens"]))
        self.num_ctx_entry = self.add_labeled_entry("Контекст (num_ctx)", str(self.config["num_ctx"]))
        self.repeat_penalty_entry = self.add_labeled_entry("Штраф за повторения", str(self.config["repeat_penalty"]))
        self.seed_entry = self.add_labeled_entry("Seed (оставьте пустым для случайного)", str(self.config["seed"]))

        self.file_label = ctk.CTkLabel(self, text="Файл не выбран")
        self.file_label.pack(pady=5)
        self.choose_button = ctk.CTkButton(self, text="Выбрать CSV", command=self.choose_file)
        self.choose_button.pack(pady=5)
        self.run_button = ctk.CTkButton(self, text="Запустить", command=self.run)
        self.run_button.pack(pady=10)
        self.status_label = ctk.CTkLabel(self, text="")
        self.status_label.pack(pady=5)

    def add_labeled_entry(self, label_text, default_value):
        label = ctk.CTkLabel(self, text=label_text)
        label.pack(pady=(10, 0))
        entry = ctk.CTkEntry(self)
        entry.insert(0, default_value)
        entry.pack()
        return entry

    def choose_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.file_path = file
            self.file_label.configure(text=os.path.basename(file))

    def collect_config(self):
        try:
            return {
                "api_url": DEFAULT_CONFIG["api_url"],
                "model": self.model_entry.get(),
                "temperature": float(self.temperature_entry.get()),
                "top_p": float(self.top_p_entry.get()),
                "max_tokens": int(self.max_tokens_entry.get()),
                "num_ctx": int(self.num_ctx_entry.get()),
                "repeat_penalty": float(self.repeat_penalty_entry.get()),
                "seed": int(self.seed_entry.get()) if self.seed_entry.get().strip() else None,
                "timeout": DEFAULT_CONFIG["timeout"],
            }
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверные параметры: {str(e)}")
            return None

    def generate_text(self, prompt, context, config):
        full_prompt = f"{prompt} {context}"
        try:
            response = requests.post(
                config["api_url"],
                json={
                    "model": config["model"],
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": config["temperature"],
                        "top_p": config["top_p"],
                        "num_ctx": config["num_ctx"],
                        "repeat_penalty": config["repeat_penalty"],
                        "seed": config["seed"],
                    },
                    "num_predict": config["max_tokens"],
                },
                timeout=config["timeout"]
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            return f"Ошибка запроса: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"

    def process_csv(self):
        config = self.collect_config()
        if not config:
            return

        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            self.status_label.configure(text=f"Ошибка чтения CSV: {str(e)}")
            return

        self.status_label.configure(text=f"Обработка {len(df)} строк...")
        try:
            df["result"] = df.apply(lambda row: self.generate_text(row["prompt"], row["context"], config), axis=1)
            output_path = os.path.join(os.path.dirname(self.file_path), "output.csv")
            df.to_csv(output_path, index=False)
            self.status_label.configure(text=f"Готово! output.csv")
            try:
                os.startfile(output_path)
            except:
                os.system(f"open '{output_path}'")
        except Exception as e:
            self.status_label.configure(text=f"Ошибка обработки: {str(e)}")

    def run(self):
        if not self.file_path:
            self.status_label.configure(text="Выберите CSV‑файл.")
            return
        threading.Thread(target=self.process_csv).start()

if __name__ == "__main__":
    app = LLMApp()
    app.mainloop()
