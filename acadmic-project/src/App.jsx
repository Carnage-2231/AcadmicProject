import React, { useState } from 'react';
import LivePrediction from './components/LivePrediction';
import DatasetCreator from './components/DatasetCreator';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('live');

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Sign Language Recognition System</h1>
        <p className="subtitle">Real-time hand gesture recognition with AI</p>
      </header>

      <nav className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          Live Prediction
        </button>
        <button 
          className={`tab-button ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataset')}
        >
          Create Dataset
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'live' && <LivePrediction />}
        {activeTab === 'dataset' && <DatasetCreator />}
      </main>

      <footer className="app-footer">
        <p>Built with MediaPipe, PyTorch & React</p>
      </footer>
    </div>
  );
}

export default App;