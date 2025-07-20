

# AMR-ICP Algorithm  🚀
**Iterative Closest Point (ICP) for Autonomous Mobile Robot Localization and Mapping**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-yellowgreen)](https://www.python.org/)

---

## 📖 Overview  
This repository implements an optimized **Iterative Closest Point (ICP)** algorithm for **Autonomous Mobile Robots (AMRs)**, designed for accurate and efficient LiDAR/point cloud registration in localization and mapping applications.

---

## Key Features  
- High-accuracy point cloud alignment with adaptive thresholding  
- Low computational cost (supports CPU/GPU acceleration)  
- ROS-compatible interface (optional)  

---

## Installation  

### Dependencies  
- Python 3.6+  
- NumPy, SciPy  
- Open3D or PCL  

### Steps  
```bash
git clone https://github.com/Ravindu-Kuruppuarachchi/AMR-ICP_Algorithm.git
cd AMR-ICP_Algorithm
pip install -r requirements.txt
```

---

## Usage  

### Basic Example  
```python
from amr_icp import ICP

icp = ICP(max_iterations=100, tolerance=1e-5)
aligned_cloud = icp.fit(source_cloud, target_cloud)
```



## Contributing  
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/improvement`)  
3. Commit changes (`git commit -m "Description"`)  
4. Push to branch (`git push origin feature/improvement`)  
5. Open a Pull Request  

---

## License  
MIT License - See [LICENSE](LICENSE)  

---

## Contact  
**Ravindu Kuruppuarachchi**  
- Email: [ravindukrashmika@gmail.com](mailto:ravindukrashmika@gmail.com)  
- GitHub: [Ravindu-Kuruppuarachchi](https://github.com/Ravindu-Kuruppuarachchi)  

--- 
