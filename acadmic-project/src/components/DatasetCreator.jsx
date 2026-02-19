import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Camera, CameraOff, Trash2, Play } from 'lucide-react';

const DatasetCreator = () => {
    const [gestureName, setGestureName] = useState('');
    const [numSamples, setNumSamples] = useState(20);
    const [framesPerSample, setFramesPerSample] = useState(30);
    const [existingGestures, setExistingGestures] = useState([]);
    const [isCameraOn, setIsCameraOn] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [currentSample, setCurrentSample] = useState(0);
    const [recordingProgress, setRecordingProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [countdown, setCountdown] = useState(null);
    const [videoUrl, setVideoUrl] = useState('/api/video_feed');

    const isCancelled = useRef(false);
    const retryLimit = 3;

    /* ---------------- LOAD & REFRESH GESTURES ---------------- */

    const refreshGestures = async () => {
        try {
            const response = await axios.get('/api/dataset/gestures');
            setExistingGestures(response.data.gestures || []);
        } catch (err) {
            console.error('Error loading gestures:', err);
        }
    };

    useEffect(() => {
        (async () => {
            await refreshGestures();
        })();
    }, []);

    /* ---------------- CAMERA CONTROL ---------------- */

    const toggleCamera = async () => {
        try {
            if (isCameraOn) {
                await axios.post('/api/camera/stop');
                setIsCameraOn(false);
            } else {
                await axios.post('/api/camera/start');
                setVideoUrl(`/api/video_feed?ts=${Date.now()}`);
                setIsCameraOn(true);
            }
        } catch (err) {
            console.error('Camera toggle error:', err);
            alert('Failed to toggle camera');
        }
    };

    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    /* ---------------- RECORDING LOGIC ---------------- */

    const startRecording = async () => {
        if (!gestureName.trim()) {
            alert('Please enter a gesture name');
            return;
        }

        if (!isCameraOn) {
            alert('Please turn on the camera first');
            return;
        }

        try {
            isCancelled.current = false;
            setIsRecording(true);
            setRecordingProgress(0);
            setCurrentSample(0);

            await axios.post('/api/dataset/create', {
                gesture_name: gestureName.trim(),
                num_samples: numSamples,
                frames_per_sample: framesPerSample
            });

            for (let i = 0; i < numSamples; i++) {
                if (isCancelled.current) break;

                setCurrentSample(i + 1);

                // Countdown
                for (let c = 1; c > 0; c--) {
                    setCountdown(c);
                    setStatusMessage(`Get ready... ${c}`);
                    await delay(1000);
                }

                setCountdown(null);
                setStatusMessage(`Recording sample ${i + 1}/${numSamples}...`);

                let retries = 0;
                let success = false;

                while (retries < retryLimit && !success) {
                    try {
                        await axios.post('/api/dataset/record_sample', {
                            gesture_name: gestureName.trim(),
                            sample_index: i,
                            frames_required: framesPerSample
                        });
                        success = true;
                    } catch (err) {
                        retries++;
                        console.error(`Retry ${retries} for sample ${i + 1}`, err);
                        await delay(800);
                    }
                }

                if (!success) {
                    setStatusMessage(
                        `Failed sample ${i + 1} after ${retryLimit} retries.`
                    );
                    break;
                }

                setRecordingProgress(((i + 1) / numSamples) * 100);
                await delay(800);
            }

            setIsRecording(false);
            setCountdown(null);
            setRecordingProgress(100);
            setStatusMessage(`✅ Successfully recorded ${numSamples} samples!`);

            await refreshGestures();

            setTimeout(() => {
                setRecordingProgress(0);
                setCurrentSample(0);
                setStatusMessage('');
            }, 1000);

        } catch (err) {
            console.error('Recording error:', err);
            setIsRecording(false);
            alert('Failed to start recording');
        }
    };

    const stopRecording = () => {
        isCancelled.current = true;
        setIsRecording(false);
        setStatusMessage('Recording stopped.');
    };

    /* ---------------- DELETE GESTURE ---------------- */

    const deleteGesture = async (gesture) => {
        if (!window.confirm(`Delete "${gesture}"?`)) return;

        try {
            await axios.post('/api/dataset/delete', { gesture_name: gesture });
            setStatusMessage(`Deleted "${gesture}"`);
            await refreshGestures();
            setTimeout(() => setStatusMessage(''), 3000);
        } catch (err) {
            console.error('Delete error:', err);
            alert('Failed to delete gesture');
        }
    };

    /* ---------------- UI ---------------- */

    return (
        <div className="dataset-creator-container">
            <div className="creator-panel">
                <h2>Dataset Creation</h2>

                <div className="form-group">
                    <label>Gesture Name</label>
                    <input
                        className='input-field' 
                        type="text"
                        value={gestureName}
                        onChange={(e) => setGestureName(e.target.value)}
                        disabled={isRecording}
                    />
                </div>

                <div className="form-row">
                    <input 
                        className='input-field'
                        type="number"
                        min="5"
                        max="100"
                        placeholder='Number of Samples'
                        value={numSamples}
                        onChange={(e) =>
                            setNumSamples(parseInt(e.target.value) || 0)
                        }
                        disabled={isRecording}
                    />

                    <input
                        className='input-field'
                        type="number"
                        min="15"
                        max="60"
                        placeholder='Frames per Sample'
                        value={framesPerSample}
                        onChange={(e) =>
                            setFramesPerSample(parseInt(e.target.value) || 0)
                        }
                        disabled={isRecording}
                    />
                </div>

                <div className="button-group ">
                    <button 
                    onClick={toggleCamera} 
                    disabled={isRecording} 
                    className='control-button'>
                        {isCameraOn ? <CameraOff size={18} /> : <Camera size={18} />}
                        {isCameraOn ? ' Stop Camera' : ' Start Camera'}
                    </button>

                    {!isRecording ? (
                        <button
                            className='control-button'
                            onClick={startRecording}
                            disabled={!isCameraOn || !gestureName.trim()}
                        >
                            <Play size={18} /> Start Recording
                        </button>
                    ) : (
                        <button
                         onClick={stopRecording}
                          className='control-button'>
                            Stop Recording
                        </button>
                    )}
                </div>

                {statusMessage && <p>{statusMessage}</p>}
                {countdown && <h1>{countdown}</h1>}

                {isRecording && (
                    <div>
                        <p>
                            Sample {currentSample}/{numSamples}
                        </p>
                        <div className="progress-bar">
                            <div
                                className="progress-fill"
                                style={{ width: `${recordingProgress}%` }}
                            />
                        </div>
                        <p>{Math.round(recordingProgress)}% Complete</p>
                    </div>
                )}

                <h3>Existing Gestures ({existingGestures.length})</h3>

                <ul>
                    {existingGestures.map((gesture, index) => (
                        <li key={index}>
                            📁 {gesture}
                            <button
                                onClick={() => deleteGesture(gesture)}
                                disabled={isRecording || isCameraOn}
                            >
                                <Trash2 size={14} />
                            </button>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="camera-panel">
                <h2>Camera Preview</h2>
                {isCameraOn ? (
                    <img
                        src={videoUrl}
                        alt="Camera Preview"
                        className="video-stream "
                    />
                ) : (
                    <div className="video-placeholder">
                        <CameraOff  size={64} color="#666"/>
                        <p>Camera is off</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default DatasetCreator;
