import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


class SmartLocFeatureEngineer:
    def __init__(self):
        self.col_map = {
            'cno': 'Carrier-to-noise density ratio (cno) [dbHz]',
            'prStdev': 'Estimated pseudorange measurement standard deviation (prStdev) [m]',
            'cpStdev': 'Estimated carrier phase measurement standard deviation (cpStdev) [cycles]',
            'doStdev': 'Estimated Doppler measurement standard deviation (doStdev) [Hz]',
            'elevation': 'Elevation',
            'azimuth': 'Azimuth',
            'label': 'NLOS (0 == no, 1 == yes, # == No Information)',
            'week': 'GPSWeek [weeks]',
            'tow': 'GPSSecondsOfWeek [s]',
            'gnssId': 'GNSS identifier (gnssId) []',
            'svId': 'Satellite identifier (svId) []'
        }
        self.feature_columns = ['cno_norm', 'el_norm', 'az_sin', 'az_cos',
                                'prStdev_log', 'cpStdev_log', 'doStdev_log']
        self.scaler_cno = MinMaxScaler(feature_range=(0, 1))
        self.scaler_el = MinMaxScaler(feature_range=(0, 1))
        self.scaler_stdev = StandardScaler()

    def fit(self, df):
        """只在训练集调用"""
        print("正在拟合特征缩放器...")
        if self.col_map['cno'] in df.columns:
            self.scaler_cno.fit(df[[self.col_map['cno']]])

        el_data = df[[self.col_map['elevation']]].fillna(0)
        self.scaler_el.fit(el_data)

        stdev_cols = [self.col_map['prStdev'], self.col_map['cpStdev'], self.col_map['doStdev']]
        temp_stdev = np.log1p(df[stdev_cols].fillna(0))
        temp_stdev = temp_stdev.replace([np.inf, -np.inf], 0)
        self.scaler_stdev.fit(temp_stdev)
        print("拟合完成。")

    def transform(self, df):
        """应用变换"""
        df_out = df.copy()
        # ... (此处代码逻辑与你原版一致，省略重复部分，重点是下面的保存/加载) ...
        # --- 为了节省篇幅，这里请保留你原版 transform 的完整代码 ---

        # 复制你原来的 transform 代码到这里
        # ...

        # --- 下面是原版代码的 transform 内容复现 (精简版) ---
        for key, col_name in self.col_map.items():
            if col_name not in df_out.columns: df_out[col_name] = 0

        df_out['el_norm'] = self.scaler_el.transform(df_out[[self.col_map['elevation']]].fillna(0))

        az_rad = np.radians(df_out[self.col_map['azimuth']].fillna(0))
        df_out['az_sin'] = np.sin(az_rad)
        df_out['az_cos'] = np.cos(az_rad)

        df_out['cno_norm'] = self.scaler_cno.transform(df_out[[self.col_map['cno']]].fillna(0))

        stdev_cols = [self.col_map['prStdev'], self.col_map['cpStdev'], self.col_map['doStdev']]
        stdev_log = np.log1p(df_out[stdev_cols].fillna(0)).replace([np.inf, -np.inf], 0)
        stdev_scaled = self.scaler_stdev.transform(stdev_log)

        df_out['prStdev_log'] = stdev_scaled[:, 0]
        df_out['cpStdev_log'] = stdev_scaled[:, 1]
        df_out['doStdev_log'] = stdev_scaled[:, 2]

        df_out['label'] = pd.to_numeric(df_out[self.col_map['label']], errors='coerce')

        final_cols = [self.col_map['week'], self.col_map['tow'], self.col_map['gnssId'], self.col_map['svId']] + \
                     self.feature_columns + ['label']
        return df_out[final_cols]

    def save_scalers(self, path='scalers.pkl'):
        """保存拟合好的参数"""
        joblib.dump({
            'cno': self.scaler_cno,
            'el': self.scaler_el,
            'stdev': self.scaler_stdev
        }, path)
        print(f"✅ Scalers 已保存至 {path}")

    def load_scalers(self, path='scalers.pkl'):
        """加载参数"""
        data = joblib.load(path)
        self.scaler_cno = data['cno']
        self.scaler_el = data['el']
        self.scaler_stdev = data['stdev']
        print(f"✅ Scalers 已从 {path} 加载")


# --- 测试代码 ---
if __name__ == "__main__":
    csv_file = 'RXM-RAWX_with_Angle.csv'
    try:
        df = pd.read_csv(csv_file, sep=';')
        print("成功读取 CSV 文件。")

        engineer = SmartLocFeatureEngineer()
        engineer.fit(df)
        df_processed = engineer.transform(df)

        print("\n处理后的数据预览：")
        print(df_processed.head())
        print("\n数据统计信息：")
        print(df_processed.describe())

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
    except Exception as e:
        print(f"发生其他错误: {e}")