import React, { useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area } from 'recharts';

const ComplaintTrackingChart = ({ analyzedData }) => {
  // Cyberpunk color palette with green and black focus
  const colors = {
    neon: {
      green: '#00ff41',      // Matrix green
      brightGreen: '#39ff14', // Bright lime green
      darkGreen: '#008f11',  // Dark matrix green
      accent: '#0abdc6',     // Subtle cyan accent
      yellow: '#caff70'      // Subtle yellow highlight
    },
    dark: {
      base: '#0d0208',       // Almost black
      secondary: '#0f100f',  // Very dark gray
      accent: '#121212'      // Dark charcoal
    },
    text: {
      primary: '#00ff41',    // Matrix green
      secondary: '#465b49'   // Muted green
    }
  };
  
  // Prepare data for visualization - moved useMemo before early return
  const data = useMemo(() => {
    if (!analyzedData || !analyzedData.chunks || analyzedData.chunks.length === 0) return [];
    
    console.log('Chunks received in ComplaintTrackingChart:', analyzedData.chunks);
    
    // Debug information about the structure of each chunk
    analyzedData.chunks.forEach((chunk, index) => {
      console.log(`Chunk ${index}:`, {
        id: chunk.id,
        isComplaint: chunk.isComplaint,
        confidence: chunk.confidence,
        windowStart: chunk.windowStart,
        windowEnd: chunk.windowEnd,
        utterancesCount: chunk.utterances?.length
      });
    });
    
    return analyzedData.chunks.map((chunk, index) => {
      // Extract chunk number from id (e.g., "chunk-3" -> 3)
      const chunkNum = parseInt(chunk.id.split('-')[1]);
      
      // Use window start/end info if available, otherwise fallback to calculated values
      const windowStart = chunk.windowStart || chunkNum;
      const windowEnd = chunk.windowEnd || (chunkNum + 3); // Default chunk size is 4
      
      return {
        // Use windowStart as the window identifier to ensure proper display order
        window: `Window ${index + 1}`,
        // Original chunk id for debugging
        originalId: chunk.id, 
        complaintLevel: chunk.isComplaint ? (chunk.confidence > 0.7 ? 2 : 1) : 0,
        complaintPercentage: chunk.confidence * 100,
        timeRange: `${windowStart}-${windowEnd}`,
        description: chunk.isComplaint 
          ? `Caller complaint (${(chunk.confidence * 100).toFixed(0)}%)`
          : 'No complaint detected'
      };
    });
  }, [analyzedData]);
  
  // Exit early if no data - moved after useMemo
  if (!analyzedData || !analyzedData.chunks || analyzedData.chunks.length === 0) {
    return (
      <div className="flex flex-col items-center w-full p-4 rounded-lg shadow-2xl border border-green-500" style={{
        background: `linear-gradient(135deg, ${colors.dark.base} 0%, ${colors.dark.secondary} 100%)`,
        boxShadow: `0 0 20px ${colors.neon.darkGreen}`
      }}>
        <h2 className="text-2xl font-bold mb-6 font-mono uppercase tracking-widest" style={{color: colors.neon.green}}>TERMINAL // NO DATA AVAILABLE</h2>
      </div>
    );
  }
  
  // Calculate the average complaint percentage
  const totalPercentage = data.reduce((sum, item) => sum + item.complaintPercentage, 0);
  const averagePercentage = (totalPercentage / data.length).toFixed(1);
  
  // Find the peak window (highest complaint percentage)
  const peakWindow = [...data].sort((a, b) => b.complaintPercentage - a.complaintPercentage)[0]?.window || 'NONE';
  
  // Calculate runtime from first to last chunk timestamp if available
  let runtime = 'N/A';
  if (analyzedData.chunks.length > 1 && analyzedData.chunks[0].timestamp && analyzedData.chunks[analyzedData.chunks.length - 1].timestamp) {
    const firstTime = new Date(analyzedData.chunks[0].timestamp).getTime();
    const lastTime = new Date(analyzedData.chunks[analyzedData.chunks.length - 1].timestamp).getTime();
    const diffSeconds = Math.floor((lastTime - firstTime) / 1000);
    const minutes = Math.floor(diffSeconds / 60);
    const seconds = diffSeconds % 60;
    runtime = `${minutes}M:${seconds}S`;
  }

  return (
    <div className="flex flex-col items-center w-full p-3 rounded-lg shadow-2xl border border-green-500" style={{
      background: `linear-gradient(135deg, ${colors.dark.base} 0%, ${colors.dark.secondary} 100%)`,
      boxShadow: `0 0 20px ${colors.neon.darkGreen}`
    }}>
      <h2 className="text-lg font-bold mb-3 font-mono uppercase tracking-widest" style={{color: colors.neon.green, fontSize: '1.4rem', textShadow: `0 0 5px ${colors.neon.green}`}}>TERMINAL // CUSTOMER COMPLAINT MATRIX</h2>
      
      <div className="w-full mb-4 border-2 border-green-500 rounded-lg p-4" style={{
        background: 'rgba(0, 20, 0, 0.7)', 
        boxShadow: `0 0 20px ${colors.neon.green}`
      }}>
        <ResponsiveContainer width="100%" height={500}>
          <ComposedChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.2)" />
            <XAxis 
              dataKey="window" 
              angle={-45} 
              textAnchor="end" 
              height={60} 
              tick={{ fontSize: 14, fill: colors.text.primary, fontFamily: 'monospace', fontWeight: 'bold' }}
              stroke={colors.text.secondary}
            />
            <YAxis 
              yAxisId="left" 
              label={{ value: 'CMPLNT LVL', angle: -90, position: 'insideLeft', fill: colors.neon.green, fontSize: 14, fontFamily: 'monospace', fontWeight: 'bold' }} 
              tick={{ fontSize: 14, fill: colors.text.primary, fontFamily: 'monospace' }}
              stroke={colors.text.secondary}
            />
            <YAxis 
              yAxisId="right" 
              orientation="right" 
              label={{ value: 'PCT %', angle: 90, position: 'insideRight', fill: colors.neon.green, fontSize: 14, fontFamily: 'monospace', fontWeight: 'bold' }} 
              tick={{ fontSize: 14, fill: colors.text.primary, fontFamily: 'monospace' }}
              stroke={colors.text.secondary}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: 'rgba(0, 20, 0, 0.9)', 
                borderColor: colors.neon.green,
                color: colors.neon.green,
                boxShadow: `0 0 10px ${colors.neon.green}`,
                borderRadius: '0px',
                fontFamily: 'monospace',
                fontSize: '14px',
                padding: '10px'
              }}
              formatter={(value, name) => {
                if (name === "complaintLevel") {
                  const levels = ["None", "Mild", "Strong"];
                  return [levels[value], "LEVEL"];
                }
                return [value + "%", "RATIO"];
              }}
              labelFormatter={(value, items) => {
                const item = data.find(d => d.window === value);
                return `${value} (${item.timeRange})`;
              }}
            />
            <Legend 
              wrapperStyle={{
                color: colors.neon.green,
                fontFamily: 'monospace',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
            />
            <Bar 
              yAxisId="left"
              dataKey="complaintLevel" 
              fill={colors.neon.darkGreen}
              name="Complaint Level"
              barSize={40}
              animationDuration={1500}
              style={{filter: `drop-shadow(0 0 8px ${colors.neon.darkGreen})`}}
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="complaintPercentage" 
              stroke={colors.neon.green}
              name="Complaint %"
              strokeWidth={3}
              dot={{ fill: colors.neon.green, r: 5, strokeWidth: 0 }}
              activeDot={{ fill: colors.neon.brightGreen, r: 8, strokeWidth: 0 }}
              animationDuration={1500}
              style={{filter: `drop-shadow(0 0 8px ${colors.neon.green})`}}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="w-full bg-gray-900 p-3 rounded-lg shadow-lg mb-4 border-2 border-green-500" style={{
        background: 'rgba(0, 20, 0, 0.7)', 
        boxShadow: `0 0 20px ${colors.neon.green}`
      }}>
        <h3 className="text-md font-mono font-semibold mb-2 text-center uppercase tracking-wider" style={{color: colors.neon.green, fontSize: '1rem'}}>// SYSTEM DIAGNOSTICS //</h3>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-black p-2 rounded-lg shadow-lg border-2 border-green-500 flex flex-col items-center justify-center" style={{
            boxShadow: `0 0 15px ${colors.neon.darkGreen}`,
            minHeight: '70px'
          }}>
            <span className="text-xs font-mono text-gray-400 uppercase mb-1" style={{fontSize: '0.8rem'}}>CMPLNT.RATIO</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.brightGreen, textShadow: `0 0 5px ${colors.neon.green}`}}>{averagePercentage}%</span>
          </div>
          <div className="bg-black p-2 rounded-lg shadow-lg border-2 border-green-500 flex flex-col items-center justify-center" style={{
            boxShadow: `0 0 15px ${colors.neon.darkGreen}`,
            minHeight: '70px'
          }}>
            <span className="text-xs font-mono text-gray-400 uppercase mb-1" style={{fontSize: '0.8rem'}}>PEAK.ALARM</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.brightGreen, textShadow: `0 0 5px ${colors.neon.green}`}}>{peakWindow}</span>
          </div>
          <div className="bg-black p-2 rounded-lg shadow-lg border-2 border-green-500 flex flex-col items-center justify-center" style={{
            boxShadow: `0 0 15px ${colors.neon.darkGreen}`,
            minHeight: '70px'
          }}>
            <span className="text-xs font-mono text-gray-400 uppercase mb-1" style={{fontSize: '0.8rem'}}>RUNTIME</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.brightGreen, textShadow: `0 0 5px ${colors.neon.green}`}}>{runtime}</span>
          </div>
        </div>
      </div>
      
      <div className="w-full text-center mt-4 font-mono" style={{color: colors.neon.green, fontSize: '1rem'}}>
        <p>CORP//MAINFRAME ACCESS [v3.5.2] © 2025 [RESTRICTED]</p>
      </div>
    </div>
  );
};

export default ComplaintTrackingChart; 