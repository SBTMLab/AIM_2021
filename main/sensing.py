import RPi.GPIO as GPIO
import spidev
import time
import numpy as np

# 형광분석법, 마스크먼지 측정
class sensing():
    def __init__(self, ultraviolet_pin:int, visible_pin:int, light_channel=0):
        self.ultraviolet_pin = ultraviolet_pin
        self.visible_pin = visible_pin
        self.light_channel = light_channel

        self.spi = spidev.SpiDev()
        self.spi.open(0,0)
        self.spi.max_speed_hz = 1350000

        self.dust_before = 0
        self.dust_after = 0

    def turn_LED_on(self, sec, brightness, visible=False, freq=1000.0):
        if visible==True:
            LED_pin = self.visible_pin
        else:
            LED_pin = self.ultraviolet_pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_pin, GPIO.OUT)

        pwm = GPIO.PWM(LED_pin, freq)
        pwm.start(brightness) # 0.0~100.0
        
        time.sleep(sec)

        pwm.stop()
        GPIO.cleanup()

    def analog_read(self):
        r = self.spi.xfer2([1, (8+self.light_channel) << 4,0])
        adc_out = ((r[1]&3)<<8) + r[2]
        return adc_out

    def light_sensor(self, accumulate_time:int):
        data_list = list()
        for _ in range(accumulate_time):
            data = self.analog_read()
            data_list.append(data)
            time.sleep(1)
        data_array = np.array(data_list)
        return np.median(data_array)

    # light_sensor 함수를 이용해서 self.dust_before와 self.dust_after에 값을 준 뒤에 실행!
    def dust_variance(self):
        return self.dust_after - self.dust_before

    def dust_clear(self):
        self.dust_before = 0
        self.dust_after = 0

if __name__ == '__main__':
    rasp = sensing(ultraviolet_pin=18, visible_pin=17, light_channel=0)
    rasp.turn_LED_on(sec=100, brightness=70, visible=True)
    # print(rasp.light_sensor(accumulate_time=20))