import csv
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Raspberry Pi-specific imports
import board
import busio
import numpy as np
# Adafruit sensors
import adafruit_lsm9ds1
import adafruit_apds9960.apds9960
import adafruit_bme680
import adafruit_gps

# Constants
MAX_RETRIES = 5
RETRY_DELAY = 0.5


class SensorSuite:
    def __init__(self, i2c: busio.I2C):
        """Initializes the SensorSuite with the necessary sensors."""
        self.sensors = self.init_sensors(i2c)
        self.configure_sensors()

    def init_sensors(self, i2c: busio.I2C) -> dict:
        """Initializes all sensors and returns them as a dictionary."""
        return {
            "lsm9ds1": adafruit_lsm9ds1.LSM9DS1_I2C(i2c),
            "apds9960": adafruit_apds9960.apds9960.APDS9960(i2c),
            "bme680": adafruit_bme680.Adafruit_BME680_I2C(i2c),
            "gps": adafruit_gps.GPS_GtopI2C(i2c, debug=False)
        }

    def configure_sensors(self) -> None:
        """Configures sensor settings."""
        self.sensors["apds9960"].enable_proximity = True
        self.sensors["apds9960"].enable_color = True

    def safe_read(self, func, retries=MAX_RETRIES) -> Optional[Tuple]:
        """Safely read sensor data with retries and graceful error handling."""
        for attempt in range(1, retries + 1):
            try:
                return func()
            except Exception as e:
                print(f"⚠️ Error reading sensor: {e} (Attempt {attempt}/{retries})")
                time.sleep(RETRY_DELAY)
        print("❌ Failed to read sensor data after retries.")
        return None

    def read_lsm9ds1(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], float]]:
        """Reads LSM9DS1 acceleration, gyroscope, magnetometer, and temperature with map conversion."""
        result = self.safe_read(lambda: (
            tuple(self.sensors["lsm9ds1"].acceleration),   # Convert map to tuple
            tuple(self.sensors["lsm9ds1"].gyro),           # Convert map to tuple
            tuple(self.sensors["lsm9ds1"].magnetic),       # Convert map to tuple
            self.sensors["lsm9ds1"].temperature
        ))
        return result

    def read_apds9960(self) -> Optional[Tuple[int, Tuple[int, int, int, int], int]]:
        """Reads proximity and color data from the APDS9960 sensor with map conversion."""
        result = self.safe_read(lambda: (
            self.sensors["apds9960"].proximity,
            tuple(self.sensors["apds9960"].color_data),     # Convert map to tuple
            sum(self.sensors["apds9960"].color_data)
        ))
        return result

    def read_bme680(self) -> Optional[Tuple[float, float, float, float]]:
        """Reads temperature, gas resistance, humidity, and pressure with error handling."""
        return self.safe_read(lambda: (
            self.sensors["bme680"].temperature,
            self.sensors["bme680"].gas,
            self.sensors["bme680"].humidity,
            self.sensors["bme680"].pressure
        ))

    def read_gps(self) -> Optional[Tuple[float, float, float]]:
        """Reads data from the GPS module with error handling."""
        def get_gps_data():
            self.sensors["gps"].update()
            return self.sensors["gps"].latitude, self.sensors["gps"].longitude, self.sensors["gps"].speed_knots

        return self.safe_read(get_gps_data)

    def record_data(self, csv_path: str, duration: int, frequency: int = 1) -> None:
        """Records sensor data to a CSV file with error handling."""
        end_time = datetime.now() + timedelta(seconds=duration)

        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z',
                'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z',
                'Temp_LSM9DS1', 'Proximity', 'Color_R', 'Color_G', 'Color_B', 'Color_C',
                'BME680_Temp', 'Gas', 'Humidity', 'Pressure'
                #'GPS_Latitude', 'GPS_Longitude', 'GPS_Speed'
            ])

            while datetime.now() < end_time:
                timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

                # Safely read sensor data
                lsm_data = self.read_lsm9ds1()
                apds_data = self.read_apds9960()
                bme_data = self.read_bme680()
                gps_data = self.read_gps()

                # Handle missing data with placeholders
                accel, gyro, mag, temp = lsm_data or ((0, 0, 0), (0, 0, 0), (0, 0, 0), 0)
                prox, color, lux = apds_data or (0, (0, 0, 0, 0), 0)
                bme_temp, gas, humidity, pressure = bme_data or (0, 0, 0, 0)
                # lat, lon, speed = gps_data or (None, None, None)
                field_strength = np.sqrt(mag[0]**2 + mag[1]**2 + mag[2]**2)


                # Write data to CSV
                writer.writerow([
                    timestamp, *accel, *gyro, *mag, temp,
                    prox, *color, bme_temp, gas, humidity, pressure,
                    #lat, lon, speed, field_strength
                ])

                # Print sensor data for debugging
                print(f"Timestamp: {timestamp}")
                print(f"Accel: {accel}, Gyro: {gyro}, Mag: {mag}, Temp: {temp}")
                print(f"Field Magnitude: {field_strength}")
                print(f"BME680 Temp: {bme_temp}, Gas: {gas}, Humidity: {humidity}, Pressure: {pressure}")
               # print(f"GPS Latitude: {lat}, Longitude: {lon}, Speed: {speed}")
                print(f"Proximity: {prox}, Color: {color}, Lux: {lux}")
                print("-" * 40)

                time.sleep(1 / frequency)

class SensorRecorder:
    def __init__(self):
        """Initializes the main application."""
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor_suite = SensorSuite(self.i2c)
        except Exception as e:
            print(f"🔥 Error initializing: {e}")
            self.sensor_suite = None

    def start_recording(self):
        """Starts the data recording process."""
        if self.sensor_suite:
            csv_path = f"data/sensor_data_{datetime.now().strftime('%Y-%m-%dT%H:%M')}.csv"
            print("Starting data recording...")
            self.sensor_suite.record_data(csv_path, duration=20, frequency=1)
            print(f"✅ Data saved to {csv_path}")
        else:
            print("❌ Unable to start recording due to sensor initialization failure.")

class Runner:
    def __init__(self):
        self.processid = None
    def main(self):
        recorder = SensorRecorder()
        recorder.start_recording()

def main():
        recorder = SensorRecorder()
        recorder.start_recording()


if __name__ == "__main__":
    recorder = SensorRecorder()
    recorder.start_recording()