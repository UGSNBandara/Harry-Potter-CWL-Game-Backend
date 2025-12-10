# Hogwarts Spell Detector Game Backend

A Harry Potter-themed spell detection game backend built with Python, Flask, and Whisper AI. This project consists of two servers: spell detection using voice recognition and wand tracking using pose detection.

## Features

- **Spell Detection**: Uses OpenAI Whisper for real-time voice recognition to detect Harry Potter spells
- **Wand Tracking**: Real-time pose estimation for wand tip tracking using MediaPipe
- **Unity Integration**: RESTful APIs and UDP protocol designed for Unity game integration
- **Multi-threaded**: Handles concurrent requests across both services

## Prerequisites

- Python 3.8 or higher
- Webcam access (for wand tracking)
- Microphone access (for spell detection)
- Windows/Linux/Mac (tested on Windows)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd spell-detector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Servers
Run the servers individually in separate terminals:

#### Terminal 1: Spell Detection Server (Port 5000)
```bash
python spell_detector_server.py
```

#### Terminal 2: Wand Tracking Server (UDP Port 5006)
```bash
python swd.py
```

Both servers will start and display their connection details.

## API Endpoints

### Spell Detection Server (http://localhost:5000)

- `POST /detect` - Start spell detection
  - Body: `{"spell": "lumos", "timeout": 10, "sync": false}`
- `GET /status` - Check detection status
- `POST /reset` - Reset detection state
- `GET /spells` - List available spells
- `GET /health` - Health check

### Wand Tracking Server (UDP localhost:5006)

Receives wand tip position as JSON:
```json
{"u": 0.5, "v": 0.3, "hand": "right", "visible": true}
```
- `u,v`: Normalized coordinates [0..1] of wand tip
- `hand`: "left" or "right" (configurable in swd.py)
- `visible`: Whether the wand tip is detected

## Available Spells

- wingardium leviosa
- alohomora
- incendio
- lumos
- nox
- reparo
- start

## Usage Examples

### Spell Detection
1. **Start detection:**
   ```bash
   curl -X POST http://localhost:5000/detect \
        -H "Content-Type: application/json" \
        -d '{"spell": "lumos", "timeout": 10}'
   ```

2. **Check status:**
   ```bash
   curl http://localhost:5000/status
   ```

### Wand Tracking
The wand tracking server sends UDP packets to `localhost:5006` with wand tip coordinates. Configure your Unity project to listen on this port for real-time wand position data.

## Unity Integration

### HTTP API (Spell Detection Server)
Use Unity's `UnityWebRequest` or a HTTP library to call the REST API.

### UDP Wand Tracking
The wand tracking server sends JSON data to UDP port 5006. Create a UDP receiver in Unity to get real-time wand tip coordinates.

Example C# code for Unity:
```csharp
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;

public class WandTracker : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning = true;

    void Start()
    {
        udpClient = new UdpClient(5006);
        receiveThread = new Thread(ReceiveData);
        receiveThread.Start();
    }

    void ReceiveData()
    {
        while (isRunning)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string json = System.Text.Encoding.UTF8.GetString(data);
                
                // Parse JSON and update wand position
                // {"u": 0.5, "v": 0.3, "hand": "right", "visible": true}
                Debug.Log($"Wand position: {json}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"UDP Receive error: {e.Message}");
            }
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        if (receiveThread != null) receiveThread.Abort();
        udpClient.Close();
    }
}
```

## Configuration

### Spell Detection
- **Whisper Model**: Change `WHISPER_MODEL` in `spell_detector_server.py` for different accuracy/speed trade-offs
- **Confidence Threshold**: Adjust `CONFIDENCE_THRESHOLD` for detection sensitivity

### Wand Tracking
- **Camera Settings**: Modify `CAM_INDEX`, `FRAME_W`, `FRAME_H` in `swd.py`
- **Hand Selection**: Change `HAND = "right"` to `"left"` in `swd.py` for left-handed wand
- **UDP Port**: Modify `SEND_ADDR` in `swd.py` if you need a different port
- **Smoothing**: Adjust `EMA_SEC` for position smoothing (0 to disable)

## Troubleshooting

- **Port conflicts**: Ensure port 5000 is available and UDP 5006
- **Microphone/Webcam issues**: Ensure permissions and drivers are working
- **Whisper errors**: Try a smaller model like "tiny" if "base" is too slow
- **MediaPipe errors**: Ensure webcam is accessible and not used by other apps

## License

This project is for educational purposes. Feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For questions or issues, please open a GitHub issue.