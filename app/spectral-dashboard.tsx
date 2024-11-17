'use client'

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, BarChart2, Target, Gauge } from 'lucide-react';

const SpectralDashboard = () => {
  const [modelMetrics, setModelMetrics] = useState({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
  });
  const [trainingHistory, setTrainingHistory] = useState({ epochs: [], accuracy: [], loss: [], valAccuracy: [], valLoss: [] });
  const [spectralData, setSpectralData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const metricsResponse = await fetch('/api/model-metrics');
        const metricsData = await metricsResponse.json();
        setModelMetrics(metricsData);

        const historyResponse = await fetch('/api/training-history');
        const historyData = await historyResponse.json();
        setTrainingHistory(historyData);

        const spectralResponse = await fetch('/api/spectral-data');
        const spectralData = await spectralResponse.json();
        setSpectralData(spectralData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="container mx-auto p-4 space-y-4">
      <h1 className="text-3xl font-bold mb-6">Spectral Classification Dashboard</h1>
      
      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Accuracy</div>
            <Activity className="h-4 w-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold">{(modelMetrics.accuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Precision</div>
            <Target className="h-4 w-4 text-green-600" />
          </div>
          <div className="text-2xl font-bold">{(modelMetrics.precision * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Recall</div>
            <BarChart2 className="h-4 w-4 text-purple-600" />
          </div>
          <div className="text-2xl font-bold">{(modelMetrics.recall * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">F1 Score</div>
            <Gauge className="h-4 w-4 text-red-600" />
          </div>
          <div className="text-2xl font-bold">{(modelMetrics.f1Score * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Training History Chart */}
      <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Training History</h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trainingHistory.epochs.map((epoch, index) => ({
              epoch,
              accuracy: trainingHistory.accuracy[index],
              loss: trainingHistory.loss[index],
              valAccuracy: trainingHistory.valAccuracy[index],
              valLoss: trainingHistory.valLoss[index],
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="epoch" stroke="#6b7280" tickLine={false} />
              <YAxis stroke="#6b7280" tickLine={false} domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#2563eb" name="Training Accuracy" dot={{ r: 4 }} strokeWidth={2} />
              <Line type="monotone" dataKey="loss" stroke="#dc2626" name="Training Loss" dot={{ r: 4 }} strokeWidth={2} />
              <Line type="monotone" dataKey="valAccuracy" stroke="#16a34a" name="Validation Accuracy" dot={{ r: 4 }} strokeWidth={2} />
              <Line type="monotone" dataKey="valLoss" stroke="#9333ea" name="Validation Loss" dot={{ r: 4 }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sample Spectra Chart */}
      <div className="bg-white rounded-2xl shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Sample Spectra</h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="wavelength" stroke="#6b7280" tickLine={false} label={{ value: 'Wavelength (Å)', position: 'bottom' }} />
              <YAxis stroke="#6b7280" tickLine={false} label={{ value: 'Flux', angle: -90, position: 'left' }} />
              <Tooltip />
              <Legend />
              {spectralData.map((spectrum, index) => (
                <Line 
                  key={index}
                  type="monotone" 
                  data={spectrum.wavelength.map((w, i) => ({ wavelength: w, flux: spectrum.flux[i] }))}
                  dataKey="flux" 
                  stroke={`hsl(${index * 60}, 70%, 50%)`} 
                  name={`Spectrum ${index + 1} (${spectrum.label})`}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default SpectralDashboard;