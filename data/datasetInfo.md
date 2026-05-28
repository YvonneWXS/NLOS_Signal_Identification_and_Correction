# TU Chemnitz smartLoc GNSS 数据集
---
数据集说明
**TU Chemnitz smartLoc 城市GNSS数据集**，包含8类标准文件，所有数据按 GPSWeek + GPSSecondsOfWeek 时间对齐：

1. **BESTPOS.csv**：NovAtel高精度定位真值，含时间、经纬度、海拔、定位精度、卫星统计、信号掩码
2. **BESTVEL.csv**：NovAtel速度真值，含水平速度、垂直速度、对地航向、时延校正
3. **NAV-POSLLH.csv**：u‑blox定位结果 + 真值位置/速度/航向/加速度/偏航率及全部协方差，含ublox自身hAcc/vAcc精度
4. **RXM-RAWX.csv**：u‑blox原始观测，含伪距、载波相位、多普勒、载噪比、卫星ID、观测标准差、**NLOS标注（0/1/#）**
5. **VELOCITY.csv**：车辆CAN速度
6. **YAWRATE.csv**：车辆CAN偏航率
7. **Camera data**：5路相机索引与图片，按GPS时间戳匹配
8. **SP3**：IGS精密卫星星历，用于计算卫星位置、仰角、方位角、几何伪距


---

# 1. BESTPOS.csv（NovAtel 参考定位真值）
**来源**：高精度 NovAtel SPAN RTK+IMU 参考接收机
**用途**：提供最高精度位置真值，用于伪距误差计算、定位结果验证、轨迹基准
**完整字段**：
1. GPSWeek：GPS周数
2. GPSSecondsOfWeek：GPS周内秒数
3. sol stat：定位解状态
4. pos type：定位类型
5. lat：纬度（deg）
6. lon：经度（deg）
7. hgt：平均海平面高度（m）
8. undulation：大地水准面差距（m）
9. datum id#：基准面ID
10. lat σ：纬度标准差（m）
11. lon σ：经度标准差（m）
12. hgt σ：高度标准差（m）
13. stn id：基准站ID
14. diff_age：差分龄期（s）
15. sol_age：解算龄期（s）
16. #SVs：跟踪卫星总数
17. #solnSVs：参与解算卫星数
18. #solnL1SVs：参与解算的L1/E1/B1卫星数
19. #solnMultiSVs：参与解算的多频卫星数
20. ext sol stat：扩展解状态
21. Galileo and BeiDou sig mask：Galileo/北斗信号使用掩码
22. GPS and GLONASS sig mask：GPS/GLONASS信号使用掩码

---

# 2. BESTVEL.csv（NovAtel 参考速度真值）
**来源**：NovAtel 参考接收机
**用途**：提供高精度速度、航向真值，用于运动约束、轨迹预测、滤波辅助
**完整字段**：
1. GPSWeek：GPS周数
2. GPSSecondsOfWeek：GPS周内秒数
3. sol status：速度解状态
4. vel type：速度类型
5. latency：速度时间戳延迟（s，需从时间中减去）
6. age：差分龄期（s）
7. hor spd：地面水平速度（m/s）
8. track over ground：对地航向（相对于真北，deg）
9. vertical speed：垂直速度（正=上升，负=下降，m/s）

---

# 3. NAV-POSLLH.csv（u‑blox 定位结果 + 真值GT）
**来源**：u‑blox M8T 低成本接收机 + 真值系统融合
**用途**：同时提供低成本定位结果与真值，用于监督学习、误差分析、对比实验
**完整字段**：
1. GPSWeek：GPS周数（ublox）
2. GPSSecondsOfWeek：GPS周内秒数（ublox）
3. GT Lon：真值经度（deg）
4. GT Lon Cov：真值经度协方差（deg）
5. GT Lat：真值纬度（deg）
6. GT Lat Cov：真值纬度协方差（deg）
7. GT Height：真值椭球高（m）
8. GT Height Cov：真值椭球高协方差（m）
9. GT Heading：真值航向（0=北，rad）
10. GT Heading Cov：真值航向协方差（rad）
11. GT Acceleration：真值加速度（m/s²）
12. GT Acceleration Cov：真值加速度协方差（m/s²）
13. GT Velocity：真值速度（m/s）
14. GT Velocity Cov：真值速度协方差（m/s）
15. GT Yawrate：真值偏航率（rad/s）
16. GT Yaw-rate Cov：真值偏航率协方差（rad/s）
17. iTOW：ublox 导航历元时间（ms）
18. lon：ublox 解算经度（deg）
19. lat：ublox 解算纬度（deg）
20. height：ublox 解算椭球高（m）
21. hMSL：ublox 解算平均海平面高度（m）
22. hAcc：水平精度估计（m）
23. vAcc：垂直精度估计（m，-1=无信息）

---

# 4. RXM-RAWX.csv（u‑blox 原始观测 + NLOS 标注）
**来源**：u‑blox 原始观测 + NovAtel NLOS 标注
**用途**：**整个NLOS识别、伪距误差建模、GAT训练的核心文件**
**完整字段**：
1. GPSWeek：GPS周数（ublox）
2. GPSSecondsOfWeek：GPS周内秒数（ublox）
3. GT Lon：真值经度（deg）
4. GT Lon Cov：真值经度协方差（deg）
5. GT Lat：真值纬度（deg）
6. GT Lat Cov：真值纬度协方差（deg）
7. GT Height：真值椭球高（m）
8. GT Height Cov：真值椭球高协方差（m）
9. GT Heading：真值航向（rad）
10. GT Heading Cov：真值航向协方差（rad）
11. GT Acceleration：真值加速度（m/s²）
12. GT Acceleration Cov：真值加速度协方差（m/s²）
13. GT Velocity：真值速度（m/s）
14. GT Velocity Cov：真值速度协方差（m/s）
15. GT Yawrate：真值偏航率（rad/s）
16. GT Yaw-rate Cov：真值偏航率协方差（rad/s）
17. rcvTow：测量时间（s）
18. week：GPS周号
19. leapS：GPS跳秒（s）
20. numMeas：后续观测数量
21. recStat：接收机跟踪状态
22. prMes：伪距观测值（m）
23. cpMes：载波相位观测值（cycles）
24. doMes：多普勒观测值（Hz）
25. gnssId：GNSS系统标识
26. svId：卫星编号
27. freqId：频点号（仅GLONASS）
28. locktime：载波相位锁定时间（ms）
29. cno：载噪比（dBHz）
30. prStdev：伪距估计标准差（m）
31. cpStdev：载波相位估计标准差（cycles）
32. doStdev：多普勒估计标准差（Hz）
33. trkStat：跟踪状态
34. NLOS：NLOS标注（0=LOS，1=NLOS，#=无信息，来自NovAtel）

---

# 5. VELOCITY.csv（车辆CAN速度）
**来源**：车载CAN总线
**用途**：提供车辆实时速度，用于运动模型、轨迹平滑、预测
**完整字段**：
1. GPSWeek：GPS周数
2. GPSSecondsOfWeek：GPS周内秒数
3. Velocity：车辆速度（m/s）

---

# 6. YAWRATE.csv（车辆CAN偏航率）
**来源**：车载CAN总线
**用途**：提供车辆转向速率，用于运动约束、姿态预测、滤波
**完整字段**：
1. GPSWeek：GPS周数
2. GPSSecondsOfWeek：GPS周内秒数
3. Yaw-rate：偏航角速度（rad/s）

---

# 7. Camera data（相机索引文件 + 图片）
**来源**：5路车载相机
**用途**：视觉辅助定位、场景验证、多模态学习
**包含文件**：
- GuppyFront.csv
- GuppyRear.csv
- AglaiaFrontLeft.csv
- AglaiaFrontRight.csv
- AglaiaRear.csv

**索引文件字段**：
1. GPSWeek：GPS周数
2. GPSSecondsOfWeek：GPS周内秒数
3. 帧编号
4. 对应图片文件名

**图片命名格式**：GPSWeek_GPSSecondsOfWeek.png

---

# 8. SP3 精密卫星星历文件
**文件名示例**：gmbGPSweekGPSDayofWeek.sp3.Z
**来源**：IGS MGEX GFZ
**用途**：精确计算卫星ECEF位置 → 计算卫星仰角、方位角、几何距离、伪距误差




