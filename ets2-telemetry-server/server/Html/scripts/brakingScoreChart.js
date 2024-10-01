// Braking Chart
const brakingCtx = document.getElementById('brakingChart').getContext('2d');
const initialBrakingData = JSON.parse(localStorage.getItem('brakingData')) || [0, 0];

const brakingChart = new Chart(brakingCtx, {
  type: 'doughnut',
  data: {
    labels: ['Smooth', 'Harsh'],
    datasets: [{
      data: initialBrakingData,
      backgroundColor: ['#045CB3', '#FF5630'],
      borderWidth: 0,
    }]
  },
  options: {
    cutout: '85%',
    responsive: true,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        enabled: false
      },
      centerText: {
        text: getDominantBraking(initialBrakingData),
        font: {
          size: '16'
        }
      }
    },
    animation: {
      duration: 1000,
    },
  }
});

// Central Text Plugin for Braking Chart
Chart.register({
  id: 'centerText',
  beforeDraw: (chart) => {
    const { ctx, chartArea: { left, top, width, height } } = chart;
    ctx.save();
    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const text = chart.options.plugins.centerText.text;
    ctx.fillText(text, left + width / 2, top + height / 2);
    ctx.restore();
  }
});

// Setup SSE connection for Braking Chart
const brakingEventSource = new EventSource('http://localhost:8080/sse');

brakingEventSource.onopen = () => {
  console.log('SSE connected for Braking Chart');
};

brakingEventSource.onerror = (error) => {
  console.error('SSE error for Braking Chart:', error);
};

brakingEventSource.onmessage = (event) => {
  handleBrakingData(event);
};

// Handle incoming data for Braking Chart
function handleBrakingData(event) {
  const record = JSON.parse(event.data);

  // Update braking chart
  const brakingData = brakingChart.data.datasets[0].data;
  if (record.braking_prediction === 'smooth') brakingData[0]++;
  if (record.braking_prediction === 'harsh') brakingData[1]++;
  localStorage.setItem('brakingData', JSON.stringify(brakingData));
  brakingChart.options.plugins.centerText.text = getDominantBraking(brakingData);
  brakingChart.update();
}

// Helper function to get dominant braking
function getDominantBraking(data) {
  if (data.reduce((a, b) => a + b, 0) === 0) {
    return 'Start driving';
  }
  const maxIndex = data.indexOf(Math.max(...data));
  return ['Smooth braking', 'Harsh braking'][maxIndex];
}

// Update the initial center text for Braking Chart
brakingChart.options.plugins.centerText.text = getDominantBraking(initialBrakingData);

// Reset button for Braking Chart
document.getElementById('resetButton').addEventListener('click', () => {
  brakingChart.data.datasets[0].data = [0, 0];
  localStorage.removeItem('brakingData');
  brakingChart.options.plugins.centerText.text = getDominantBraking([0, 0]);
  brakingChart.update();
});
