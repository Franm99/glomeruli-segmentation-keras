import time
from typing import Optional
from tkinter import filedialog
import os
from getpass import getpass
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


class EmailHandler:
    def __init__(self):
        def_sender_email = "pythonAdvisor22@gmail.com"
        req = input("Use default sender ({}) [Y/n]: ".format(def_sender_email))
        if req.lower() == "y":
            self._sender = def_sender_email
        else:
            self._sender = input("Specify sender email: ")
        self._pass = getpass()  # Just for terminal executions, not IDLE!
        self._recv = input("Specify receiver email: ")

    def send_sample_info(self, t: float, fname: str) -> None:
        time_mark = time.strftime("%H:%M:%S", time.gmtime(t))
        html = """\
                <html>
                    <body>
                        Training finished. For further info, check log file.<br>
                        Time spent: {} (h:m:s)<br>
                    </body>
                </html>
                """.format(time_mark)
        self._send_message(subject="Sample training finished", html=html, log_file=fname)

    def send_session_info(self, fname: str) -> None:
        self._send_message(subject="Session ended", log_file=fname)

    def _send_message(self, subject: str, html: Optional[str] = None, log_file: Optional[str] = None) -> None:
        port = 465
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self._sender
        message["To"] = self._recv

        if html:
            part1 = MIMEText(html, "html")
            message.attach(part1)

        if log_file:
            with open(log_file, "rb") as att:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(att.read())
            encoders.encode_base64(part)
            part.add_header("Content-disposition",
                            f"attachment; filename = {os.path.basename(log_file)}")
            message.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(self._sender, self._pass)
            server.sendmail(self._sender, self._recv, message.as_string())


def browse_path():
    """
    Opens a file browser to select the path from where test prediction images are taken.
    Default initial directory: output/ folder.

    **NOTE**: To select a certain output folder, you may first enter to that folder!
    """
    full_path = filedialog.askdirectory(initialdir='../output')
    return full_path


def browse_file():
    return filedialog.askopenfilename(initialdir='data/raw')


def timer(f):
    """
    Timer decorator to wrap and measure a function time performance.

    :param f: Function to wrap.
    :return: decorated function
    """
    def time_dec(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print_info("{} - {:2.4f} sec".format(f.__name__, te - ts))
        return result

    return time_dec


def print_info(msg):
    """ Log INFO messages """
    info = "--> [I]:  "
    print(info, msg)


def print_warn(msg):
    """ Log WARNING messages """
    info = "--> [W]:  "
    print(info, msg)


def print_error(msg):
    """ Log ERROR messages """
    info = "--> [E]:  "
    print(info, msg)


def check_gpu_availability():
    """
    Check if there are any available GPU resources.

    Source: https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
