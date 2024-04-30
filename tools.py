
import functools
import itertools
import json
import os
import sys
import time
from matplotlib import pyplot as plt

import numpy as np
import requests
import yaml

__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(__PATH__)  

# ---------------------- 单例模式 -------------------------- #
def singleton(cls):
    _instance = {}

    def inner(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]
    return inner

# ---------------------- 配置信息 -------------------------- #
@singleton
class Config:
    def __init__(self) -> None:
        with open("config.yml", "r") as config_open:
            self.data = yaml.safe_load(config_open)
    def __getitem__(self, name):
        return self.data[name]
# ---------------------- 全局变量 -------------------------- #
# 校园网账号密码
config = Config()
USERNAME = config["network"]["username"]
PASSWORD = config["network"]["password"]
WX_TOKEN = config["network"]["wx_token"]
# ---------------------- 微信通知 -------------------------- #
class Notice:
    # 校园网网关登录地址 或换成"http://gw.bupt.edu.cn/login"
    Getway_IP = "http://10.3.8.211/login"
    # 校园网网关登出地址 或换成"http://gw.bupt.edu.cn/logout"
    LogOut_URL = "http://10.3.8.211/logout"
    Check_URL = "http://www.baidu.com"          # 用以检测是否可以连接到外网
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'}

    def check(self):
        # 必须禁止重定向，否则 status_code 一直是 200
        res = requests.get(Notice.Check_URL, timeout=1, allow_redirects=False)
        if res.status_code == 200:
            return True
        else:
            return False

    def login(self, username, password):
        params = {
            'user': username,
            'pass': password
        }
        res = requests.post(
            Notice.Getway_IP, headers=Notice.headers, params=params)
        return res

    def logout(self):
        res = requests.get(Notice.LogOut_URL,
                           headers=Notice.headers, allow_redirects=False)
        return res

    def send(self, txt):
        url = 'http://wxpusher.zjiecode.com/api/send/message'
        body = {
            "appToken": WX_TOKEN,
            "uids": ["UID_tf2Vb3egqGigY58kFDsbmHovvQ0h"],
            "content": txt,
            "contentType": 1
        }
        headers = {'Content-Type': 'application/json; charset=UTF-8'}

        if not self.check():
            self.login(USERNAME, PASSWORD)
        requests.post(url, data=json.dumps(body), headers=headers)

# ---------------------- 计时器 -------------------------- #
class Timer:
    '''记录运行时间，内部维护一个耗时队列'''

    def __init__(self) -> None:
        self.times = []
        self.tik = time.time()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        return sum(self.times)

    def avg(self):
        return sum(self.times) / len(self.times)

    def cunsum(self):
        return np.array(self.times).cumsum().tolist()

    def clear(self):
        self.times.clear()

# ---------------------- 绘制图像 -------------------------- #

class Painter:
    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, path='./img/confusion_matrix.png',title='Confusion matrix', cmap=plt.cm.Blues):
        """
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        
        plt.figure ()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path)
        plt.show()

def timeout(sec, raise_sec=1):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    :param raise_sec: retry kill thread per ? seconds
        default: 1 second
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            err_msg = f'Function {func.__name__} timed out after {sec} seconds'

            if sys.platform != 'win32':
                import signal

                def _handle_timeout(signum, frame):
                    raise TimeoutError(err_msg)

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(sec)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

            if sys.platform == 'win32':
                class FuncTimeoutError(TimeoutError):
                    def __init__(self):
                        TimeoutError.__init__(self, err_msg)

                result, exception = [], []

                def run_func():
                    try:
                        res = func(*args, **kwargs)
                    except FuncTimeoutError:
                        pass
                    except Exception as e:
                        exception.append(e)
                    else:
                        result.append(res)

                # typically, a python thread cannot be terminated, use TerminableThread instead
                thread = TerminableThread(target=run_func, daemon=True)
                thread.start()
                thread.join(timeout=sec)

                if thread.is_alive():
                    # a timeout thread keeps alive after join method, terminate and raise TimeoutError
                    exc = type('TimeoutError', FuncTimeoutError.__bases__, dict(FuncTimeoutError.__dict__))
                    thread.terminate(exception_cls=FuncTimeoutError, repeat_sec=raise_sec)
                    raise TimeoutError(err_msg)
                elif exception:
                    # if exception occurs during the thread running, raise it
                    raise exception[0]
                else:
                    # if the thread successfully finished, return its results
                    return result[0]

        return wrapped_func
    return decorator