from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# dust info, KNN, visualize
class classification_model():
    def __init__(self):
        self.data = None
        self.reference = None
        self.weight = None
        self.reference_pos = None
        self.reference_neg = None
        self.result = None
        self.fine_dust = None
        self.ultra_fine_dust = None
        self.fine_dust_text = None
        self.ultra_fine_dust_text = None

    def get_dust_info(self):
        html = requests.get('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=서울특별시 서대문구 미세먼지')
        soup = bs(html.text,'html.parser')
        dust_data = str(soup.find('div',{'class':'detail_info lv2'}).findAll('dd')[0].contents[0])

        ultra_html = requests.get('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=서울특별시 서대문구 초미세먼지')
        ultra_soup = bs(ultra_html.text,'html.parser')
        ultra_dust_data = str(ultra_soup.find('div',{'class':'detail_info lv2'}).findAll('dd')[0].contents[0])
        
        '''
        self.fine_dust = dust_data[0].find('span',{'class':'num'}).text.split('㎍')[0]
        self.ultra_fine_dust = dust_data[1].find('span',{'class':'num'}).text.split('㎍')[0]

        self.fine_dust_text = dust_data[0].text.split('㎥')[-1]
        self.ultra_fine_dust_text = dust_data[1].text.split('㎥')[-1]

        return int(self.fine_dust), int(self.ultra_fine_dust), self.fine_dust_text, self.ultra_fine_dust_text
        '''

        return dust_data, ultra_dust_data

    def read_spectroscope(self, file_directory, lower_bound, upper_bound):
        read_data = pd.read_excel(file_directory, header=None, skiprows=7)
        # print(read_data)
        light_data = read_data[read_data[0] >= lower_bound]
        light_data = light_data[light_data[0] <= upper_bound]
        light_data_list = light_data[1].values.tolist()
        light_array = np.array(light_data_list)
        return light_array.mean()

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

        return self.result

    def visualize(self):
        plt.style.use('seaborn')

        self.data = self.data.squeeze()
        if self.data.ndim == 1:
            self.data = np.expand_dims(self.data, axis=0)

        self.reference_pos = self.reference_pos.squeeze()
        if self.reference_pos.ndim == 1:
            self.reference_pos = np.expand_dims(self.reference_pos, axis=0)

        self.reference_neg = self.reference_neg.squeeze()
        if self.reference_neg.ndim == 1:
            self.reference_neg = np.expand_dims(self.reference_neg, axis=0)

        cmap = cm.get_cmap('rainbow', lut=2)

        n_feature = self.weight.shape[0]

        if n_feature == 2:
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.scatter(self.reference_pos[..., 0], self.reference_pos[..., 1], color=cmap(0), alpha=0.3)
            ax.scatter(self.reference_neg[..., 0], self.reference_neg[..., 1], color=cmap(1), alpha=0.3)
            for data_idx, data in enumerate(self.data):
                ax.scatter(data[0], data[1], color=cmap(self.result[data_idx]), marker='*')

            ax.set_xlabel("Riboflavin", fontsize=15)
            ax.set_ylabel("Dust", fontsize=15)
            
            # for legend
            ax.scatter([], [], color=cmap(0), label='positive')
            ax.scatter([], [], color=cmap(1), label='negative')
            ax.scatter([], [], color='k', marker='*', label='data')
            ax.scatter([], [], color='k', marker='o', label='reference')

            ax.legend(loc='upper left',
                      bbox_to_anchor=(1, 1),
                      ncol=2)

        elif n_feature == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection='3d')
            fig.subplots_adjust(bottom=0, top=1,
                                left=0, right=1)

            ax.set_xlabel("Riboflavin", fontsize=20, labelpad=20)
            ax.set_ylabel("Dust", fontsize=20, labelpad=20)
            ax.set_zlabel("damaged", fontsize=20, labelpad=20)

            ax.scatter(self.reference_pos[..., 0], self.reference_pos[..., 1], self.reference_pos[..., 2], 
                       color=cmap(0), 
                       alpha=0.3,
                       s=50)
            ax.scatter(self.reference_neg[..., 0], self.reference_neg[..., 1], self.reference_neg[..., 2], 
                       color=cmap(1), 
                       alpha=0.3,
                       s=50)
            for data_idx, data in enumerate(self.data):
                ax.scatter(data[0], data[1], data[2], 
                           color=cmap(self.result[data_idx]), 
                           marker='*',
                           s=50)

            # for legend
            ax.scatter([], [], [], color=cmap(0), label='positive')
            ax.scatter([], [], [], color=cmap(1), label='negative')
            ax.scatter([], [], [], color='k', marker='*', label='data')
            ax.scatter([], [], [], color='k', marker='o', label='reference')

            plt.legend(loc="upper left")
        
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    reference_pos = np.random.randint(low=0, high=10, size=(20, 2))
    reference_neg = np.random.randint(low=20, high=30, size=(30, 2))
    data = np.random.randint(low=0, high=30, size=(2, ))

    model = classification_model()
    '''model.weighted_KNN(K=5, 
                       data=data, 
                       reference=[reference_pos, reference_neg], 
                       weight=np.array([1, 3]))
    model.visualize()'''
    print(model.get_dust_info())
    print(model.read_spectroscope(file_directory='20211024.xlsx', lower_bound=510, upper_bound=540))