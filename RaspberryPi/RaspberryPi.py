from bs4 import BeautifulSoup as bs
import requests
# import RPi.GPIO as GPIO
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 형광분석법, 먼지측정
class sensing():
    def __init__(self, led_pin=18):
        self.led_pin = led_pin

    '''def turn_LED_on(self, sec, brightness, freq=1000.0):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT)

        pwm = GPIO.PWM(self.led_pin, freq)
        pwm.start(brightness) # 0.0~100.0
        
        time.sleep(sec)

        pwm.stop()
        GPIO.cleanup()'''

# 날씨정보, KNN, visualize
class classification_model():
    def __init__(self):
        pass

    def get_dust_info(self):
        html = requests.get('https://search.naver.com/search.naver?query=날씨')
        soup = bs(html.text,'html.parser')

        data1 = soup.find('div',{'class':'detail_box'})
        data2 = data1.findAll('dd')

        fine_dust = data2[0].find('span',{'class':'num'}).text.split('㎍')[0]
        ultra_fine_dust = data2[1].find('span',{'class':'num'}).text.split('㎍')[0]

        return int(fine_dust), int(ultra_fine_dust)

    def weighted_KNN(self, K: int, data: np.array, reference: list, weight: np.array):
        self.data = data # shape: (n_data, n_feature)
        self.reference = reference # len(reference) == 2, shape of element: (n_reference, n_feature)
        self.weight = weight # shape: (n_feature, )

        self.reference_pos = self.reference[0]
        self.reference_neg = self.reference[1]

        # got only one data or reference
        if self.data.ndim == 1:
            self.data = np.expand_dims(self.data, axis=0)
        if self.reference_pos.ndim == 1:
            self.reference_pos = np.expand_dims(self.reference_pos, axis=0)
        if self.reference_neg.ndim == 1:
            self.reference_neg = np.expand_dims(self.reference_neg, axis=0)

        # for broadcasting
        self.data = np.expand_dims(self.data, axis=1)
        self.reference_pos = np.expand_dims(self.reference_pos, axis=0)
        self.reference_neg = np.expand_dims(self.reference_neg, axis=0)
        
        # weighted L1 distance
        distance_pos = np.abs(self.data - self.reference_pos) * self.weight
        distance_neg = np.abs(self.data - self.reference_neg) * self.weight
        distance_total = np.concatenate([distance_pos, distance_neg], axis=1).sum(axis=-1)

        # distance sorting & classification
        distance_argsort = np.argsort(distance_total, axis=-1)[..., :K]
        distance_neg_cnt = (distance_argsort >= self.reference_pos.shape[1]).sum(axis=-1)
        self.result = np.where(distance_neg_cnt > (K-1)/2, 1, 0) # integer K has to be an odd number and smaller than number of references

    def visualize(self, n_feature=2):
        plt.style.use('seaborn')
        if n_feature == 2:
            self.data = self.data.squeeze()
            self.reference_pos = self.reference_pos.squeeze()
            self.reference_neg = self.reference_neg.squeeze()

            cmap = cm.get_cmap('rainbow', lut=2)

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.scatter(self.reference_pos[..., 0], self.reference_pos[..., 1], color=cmap(0), alpha=0.3)
            ax.scatter(self.reference_neg[..., 0], self.reference_neg[..., 1], color=cmap(1), alpha=0.3)
            for data_idx, data in enumerate(self.data):
                ax.scatter(data[0], data[1], color=cmap(self.result[data_idx]), marker='*')
            
            ax.scatter([], [], color=cmap(0), label='positive')
            ax.scatter([], [], color=cmap(1), label='negative')
            ax.scatter([], [], color='k', marker='*', label='data')
            ax.scatter([], [], color='k', marker='o', label='reference')

        ax.legend(loc='upper left',
                  bbox_to_anchor=(1, 1),
                  ncol=2)
        fig.tight_layout()
        plt.show()

reference_pos = np.random.randint(low=0, high=10, size=(20, 2))
reference_neg = np.random.randint(low=20, high=30, size=(20, 2))
data = np.random.randint(low=-10, high=40, size=(40, 2))

model = classification_model()
arr = model.weighted_KNN(5, data=data, reference=[reference_pos, reference_neg], weight=np.array([1, 4]))
model.visualize()
