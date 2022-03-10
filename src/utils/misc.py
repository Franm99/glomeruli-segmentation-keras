import time
from typing import Optional, List, Callable
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
    """
    Email Handler
    =============

    This class creates a email contextualized handler to send report emails generated during the training process.

    **NOTE:** I have created a gmail account with the unique purpose of sending training report emails. Its password is
    private knowledge, so if you want to use this functionality, you should specify your own gmail account and password.

    Functionalities
    ---------------

    Send sample Info
    ~~~~~~~~~~~~~~~~

    Generates a new message to notice the user that a sample training has finished. The message will contain relevant
    information about the training results.

    Send session info
    ~~~~~~~~~~~~~~~~~

    Generates a new message to notice the user that the whole current training session has finished. The message will
    contain relevant information about the best results during the session.

    """
    def __init__(self):
        """ *Class constructor* """
        def_sender_email = "pythonAdvisor22@gmail.com"
        req = input("Use default sender ({}) [Y/n]: ".format(def_sender_email))
        if req.lower() == "y":
            self._sender = def_sender_email
        else:
            self._sender = input("Specify sender email: ")
        self._pass = getpass()  # Just when running in terminal, not IDE!
        self._recv = input("Specify receiver email: ")

    def send_sample_info(self, t: float, fname: str) -> None:
        """
        Generates a new message to notice the user that a sample training has finished. The message will contain
        relevant information about the training results.

        :param t: sample timestamp (duration of the training process)
        :param fname: filename of the file to be sent within the email.
        :return: None
        """
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
        """
        Generates a new message to notice the user that the whole current training session has finished. The message
        will contain relevant information about the best results during the session.

        :param fname: filename of the file to be sent within the message.
        :return: None
        """
        self._send_message(subject="Session ended", log_file=fname)

    def _send_message(self, subject: str, html: Optional[str] = None, log_file: Optional[str] = None) -> None:
        """ *Private*

        Generic method to send a message.

        Source: `How to send messages using Python
        <https://stackabuse.com/how-to-send-emails-with-gmail-using-python/>`_

        :param subject: Subject of the message. It will first show to the user the topic of the message.
        :param html: message content. It is a string containing raw html code.
        :param log_file: If needed, a file can be added to be sent within the email.
        :return: None
        """
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


def browse_path() -> str:
    """
    Opens a file browser to select the path to a specific folder.

    **NOTE**: To select a certain output folder, you may first enter to that folder.

    :return: string with the full path to the selected folder.
    """
    full_path = filedialog.askdirectory(initialdir='../output')
    return full_path


def browse_file():
    """
    Open a file browser to select the path to a specific file.

    :return: string with the full path to the selected file.
    """
    return filedialog.askopenfilename(initialdir='data/raw')


def timer(f) -> Callable:
    """
    *Decorator*

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


def list2txt(filename: str, data: List[str]) -> None:
    """
    Function to save a list of string values into a txt file.

    :param filename: txt file full path and name.
    :param data: list containing the data to be saved.
    :return: None
    """
    with open(filename, "w") as f:
        for i in data:
            f.write(i + "\n")


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
