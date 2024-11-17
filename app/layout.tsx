import React from 'react'

export const metadata = {
  title: 'Spectral Classification Dashboard',
  description: 'A dashboard for spectral classification results',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}