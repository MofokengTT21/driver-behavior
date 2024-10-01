# Driver Behavior Analysis using Euro Truck Simulator 2

Welcome to the **Driver Behavior Analysis** project! In a world where data-driven insights are crucial for enhancing safety and performance, this innovative project leverages the immersive experience of Euro Truck Simulator 2 (ETS2) to analyze and predict driver behavior in real-time. Instead of relying on costly and time-consuming data collection from actual trucks, this project utilizes the ETS2 Telemetry Web Server to gather telemetry data while you navigate the virtual roads, making it accessible for enthusiasts and researchers alike.

The goal of this project is to demonstrate how gaming technology can intersect with machine learning to provide insights into driving behavior, showcasing how virtual environments can be utilized for serious analytical purposes.

### 🚧 Early Development Notice

This project is in its **early development** phase, and I'm actively refining and optimizing the code. The current data collection methods were primarily implemented for testing purposes, it's currently being tested in Euro Truck Simulator 2 only and dummy text is used for demo, so you may encounter areas for improvement.


## Features

- **Real-time Data Collection:** Utilize the ETS2 Telemetry Web Server to gather telemetry data every second.
- **Machine Learning Integration:** Predict driving behavior using LSTM models trained on collected data. Currently only two completed, Driver Behavior and Braking Score. Still working on three more, ECO driving, Acceleration and Potholes Handling Score.
- **Customizable Dashboard:** Visualize telemetry data and predictions through a mobile-friendly HTML5 dashboard.

## Prerequisites

- **Euro Truck Simulator 2** (version 1.15+)
- **Node.js** installed on your local machine
- Basic knowledge of JavaScript and HTML/CSS for customization

## Installation Instructions

To get started with this project, follow these steps:

1. **Download the Project:**
   - Clone the repository or download the ZIP file to your local machine.

   ```bash
   git clone https://github.com/MofokengTT21/driver-behavior
   cd driver-behavior
   ```

2. **Set Up ETS2 Telemetry Web Server:**
   - Follow the detailed setup instructions for the [ETS2 Telemetry Web Server 3.2.5](https://github.com/Funbit/ets2-telemetry-server?tab=readme-ov-file). N.B: Follow the instructions only for installation. But use the files under this repository (there are few changes being made for development purposes).
   - Run the following command to install the necessary dependencies:

    ```bash
    npm install
    ```

   - Ensure that the server is correctly installed and running on your local machine.

3. **Run the Node.js Server:**
   - In your terminal, navigate to the project directory and run the following command to start the data collection and prediction server:

   ```bash
   node server.js
   ```

4. **Access the Dashboard:**
   - Open your browser and navigate to the HTML5 mobile dashboard URL displayed by the ETS2 Telemetry Web Server.
   - Select the **Truck Monitoring Simulator** skin to visualize real-time telemetry data and driver behavior predictions. OR use this link:
   
   ```bash
       http://localhost:25555/dashboard-host.html?skin=truck-monitor&ip=localhost
   ```
Here is a screenshot of how the dashboard will look like in a browser:

![](https://github.com/MofokengTT21/driver-behavior/blob/main/screenshot.png?raw=true)
   

## Usage

Once you have set up everything, you can start driving in ETS2 while the telemetry data is collected. The Node.js server will process the data every second and update the dashboard with the latest predictions on driver behavior, presented in interactive donut charts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of Euro Truck Simulator 2, [ETS2 Telemetry Web Server](https://github.com/Funbit/ets2-telemetry-server?tab=readme-ov-file) and [ETS2 Mobile Route Advisor](https://github.com/mike-koch/ets2-mobile-route-advisor) for providing the tools to make this project possible.
- Special thanks to the open-source community for their support and inspiration.

---

Dive into the virtual world of driving and uncover the intricacies of driver behavior with this exciting project! Happy driving!
