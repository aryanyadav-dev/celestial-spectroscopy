import React from 'react'
import dynamic from 'next/dynamic'

const SpectralDashboard = dynamic(() => import('./spectral-dashboard'), { ssr: false })

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100">
      <SpectralDashboard />
    </div>
  )
}