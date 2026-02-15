import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { Camera, CameraOff, AlertCircle, CheckCircle } from 'lucide-react';

// Custom hook for system status
const useSystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get('/api/status');
      setSystemStatus(response.data);
      return response.data;
    } catch (err) {
      console.error('Error loading system status:', err);
      setError(err.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  return { systemStatus, isLoading, error, refetch: loadStatus };
};

// Custom hook for predictions
const usePredictions = (isCameraOn, modelLoaded) => {
  const [prediction, setPrediction] = useState('');
  const [framesCollected, setFramesCollected] = useState(0);
  const [framesRequired, setFramesRequired] = useState(30);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isCameraOn && modelLoaded) {
      intervalRef.current = setInterval(async () => {
        try {
          const response = await axios.get('/api/prediction/current');
          setPrediction(response.data.prediction);
          setFramesCollected(response.data.frames_collected);
          setFramesRequired(response.data.frames_required);
        } catch (error) {
          console.error('Error fetching prediction:', error);
        }
      }, 500);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isCameraOn, modelLoaded]);

  return { prediction, framesCollected, framesRequired };
};

// Custom hook for camera control
const useCamera = () => {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggleCamera = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (isCameraOn) {
        await axios.post('/api/camera/stop');
        setIsCameraOn(false);
      } else {
        await axios.post('/api/camera/start');
        setIsCameraOn(true);
      }
    } catch (err) {
      console.error('Error toggling camera:', err);
      setError('Failed to toggle camera');
    } finally {
      setIsLoading(false);
    }
  }, [isCameraOn]);

  return { isCameraOn, isLoading, error, toggleCamera };
};

// Custom hook for model operations
const useModel = (refetchStatus) => {
  const [isTraining, setIsTraining] = useState(false);
  const trainingPollRef = useRef(null);

  const reloadModel = useCallback(async () => {
    try {
      const response = await axios.post('/api/model/reload');
      alert(`${response.data.message}\nLabels: ${response.data.labels.join(', ')}`);
      refetchStatus();
    } catch (error) {
      console.error('Error reloading model:', error);
      alert('Failed to reload model');
    }
  }, [refetchStatus]);

  const trainModel = useCallback(async () => {
    if (!window.confirm('Start training the model? This may take a few minutes.')) {
      return;
    }

    try {
      const response = await axios.post('/api/model/train');
      alert(response.data.message);
      setIsTraining(true);

      // Poll training status
      trainingPollRef.current = setInterval(async () => {
        try {
          const statusResponse = await axios.get('/api/training/status');
          if (!statusResponse.data.training_in_progress) {
            clearInterval(trainingPollRef.current);
            trainingPollRef.current = null;
            setIsTraining(false);
            alert('Training complete! Click "Reload Model" to use the new model.');
            refetchStatus();
          }
        } catch (error) {
          console.error('Error polling training status:', error);
        }
      }, 2000);
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training');
      setIsTraining(false);
    }
  }, [refetchStatus]);

  useEffect(() => {
    return () => {
      if (trainingPollRef.current) {
        clearInterval(trainingPollRef.current);
      }
    };
  }, []);

  return { reloadModel, trainModel, isTraining };
};

// Main component
const LivePrediction = () => {
  const { systemStatus, isLoading: isLoadingStatus, refetch } = useSystemStatus();
  const { isCameraOn, isLoading: isCameraLoading, toggleCamera } = useCamera();
  const { reloadModel, trainModel, isTraining } = useModel(refetch);
  const { prediction, framesCollected, framesRequired } = usePredictions(
    isCameraOn,
    systemStatus?.model_loaded
  );

  // Derive showSetupGuide state from systemStatus instead of using useEffect
  const [manuallyHideGuide, setManuallyHideGuide] = useState(false);
  const shouldShowSetupGuide = !manuallyHideGuide && systemStatus && !systemStatus.ready_for_prediction;

  // Calculate progress percentage
  const progressPercentage = (framesCollected / framesRequired) * 100;

  return (
    <div className="live-prediction-container">
      <div className="controls-panel">
        <h2>Live Prediction Controls</h2>

        {/* Loading State */}
        {isLoadingStatus && (
          <div className="alert alert-info">
            <div>Loading system status...</div>
          </div>
        )}

        {/* System Status Alert */}
        {!isLoadingStatus && systemStatus && !systemStatus.ready_for_prediction && (
          <div className="alert alert-warning">
            <AlertCircle size={20} />
            <div>
              <strong>Setup Required</strong>
              <p>The system is not ready for predictions yet.</p>
              <button
                className="btn-link"
                onClick={() => setManuallyHideGuide(!manuallyHideGuide)}
                type="button"
              >
                {shouldShowSetupGuide ? 'Hide' : 'Show'} Setup Guide
              </button>
            </div>
          </div>
        )}

        {/* Setup Guide */}
        {shouldShowSetupGuide && systemStatus?.setup_steps && (
          <div className="setup-guide">
            <h3>📋 Setup Steps</h3>
            {systemStatus.setup_steps.map((step) => (
              <div key={step.step} className={`setup-step ${step.completed ? 'completed' : ''}`}>
                <div className="step-icon">
                  {step.completed ? (
                    <CheckCircle size={20} color="#10b981" />
                  ) : (
                    <span className="step-number">{step.step}</span>
                  )}
                </div>
                <div className="step-content">
                  <strong>{step.action}</strong>
                  <p>{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* System Info */}
        {systemStatus && (
          <div className="system-info">
            <h3>System Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <span className="label">Dataset Folder:</span>
                <span className={`value ${systemStatus.dataset_folder_exists ? 'success' : 'error'}`}>
                  {systemStatus.dataset_folder_exists ? '✓ Exists' : '✗ Not Found'}
                </span>
              </div>
              <div className="info-item">
                <span className="label">Model File:</span>
                <span className={`value ${systemStatus.model_file_exists ? 'success' : 'error'}`}>
                  {systemStatus.model_file_exists ? '✓ Exists' : '✗ Not Found'}
                </span>
              </div>
              <div className="info-item">
                <span className="label">Gestures:</span>
                <span className="value">{systemStatus.gesture_count || 0}</span>
              </div>
              <div className="info-item">
                <span className="label">Total Samples:</span>
                <span className="value">{systemStatus.total_samples || 0}</span>
              </div>
              <div className="info-item">
                <span className="label">Model Status:</span>
                <span className={`value ${systemStatus.model_loaded ? 'success' : 'warning'}`}>
                  {systemStatus.model_loaded ? '✓ Loaded' : '⚠ Not Loaded'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <button
          className={`control-button ${isCameraOn ? 'danger' : 'primary'}`}
          onClick={toggleCamera}
          disabled={isCameraLoading}
          type="button"
        >
          {isCameraLoading ? (
            '⏳ Loading...'
          ) : isCameraOn ? (
            <>
              <CameraOff size={20} />
              Stop Camera
            </>
          ) : (
            <>
              <Camera size={20} />
              Start Camera
            </>
          )}
        </button>

        <button
          className="control-button secondary"
          onClick={reloadModel}
          type="button"
        >
        Reload Model
        </button>

        {systemStatus && systemStatus.gesture_count >= 2 && !systemStatus.model_loaded && (
          <button
            className="control-button success"
            onClick={trainModel}
            disabled={isTraining || systemStatus.training_in_progress}
            type="button"
          >
            {isTraining || systemStatus.training_in_progress ? '⏳ Training...' : '🎯 Train Now'}
          </button>
        )}

        {/* Status Panel */}
        <div className="status-panel">
          <h3>Status</h3>
          <div className="status-item">
            <span className="label">Camera:</span>
            <span className={`value ${isCameraOn ? 'active' : ''}`}>
              {isCameraOn ? '🟢 Active' : '🔴 Inactive'}
            </span>
          </div>
          <div className="status-item">
            <span className="label">Frames:</span>
            <span className="value">
              {framesCollected} / {framesRequired}
            </span>
          </div>
          <div className="status-item">
            <span className="label">Prediction:</span>
            <span className="value prediction">
              {prediction || (systemStatus?.model_loaded ? 'Waiting...' : 'Model not loaded')}
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>

      {/* Camera Panel */}
      <div className="camera-panel">
        <h2>Camera Feed</h2>
        {isCameraOn ? (
          <div className="video-container">
            <img
              src="/api/video_feed"
              alt="Live Camera Feed"
              className="video-stream"
            />
            <div className="prediction-overlay">
              <div className="prediction-text">
                {prediction || 'No gesture detected'}
              </div>
            </div>
          </div>
        ) : (
          <div className="video-placeholder">
            <CameraOff size={64} color="#666" />
            <p>Camera is off</p>
            <p className="hint">Click "Start Camera" to begin prediction</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default LivePrediction;