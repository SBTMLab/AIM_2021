import RPi.GPIO as GPIO
import spidev
import time
import numpy as np

# 형광분석법, 먼지측정
class sensing():
    def __init__(self, led_pin=18, light_channel=1):
        self.led_pin = led_pin
        self.light_channel = light_channel

        self.spi = spidev.SpiDev()
        self.spi.open(0,0)
        self.spi.max_speed_hz = 1350000

        self.dust_before = 0
        self.dust_after = 0

    def turn_LED_on(self, sec, brightness, freq=1000.0):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT)

        pwm = GPIO.PWM(self.led_pin, freq)
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