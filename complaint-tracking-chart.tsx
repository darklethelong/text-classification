import React from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area } from 'recharts';

const ComplaintTrackingChart = () => {
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
  
  // Data for each window, with complaint levels:
  // 0 = No complaint, 1 = Mild complaint, 2 = Strong complaint
  const data = [
    { 
      window: 'Window 1',
      timeRange: '00:00:05 - 00:00:37', 
      complaintLevel: 0,
      complaintPercentage: 0,
      description: 'Initial contact, issue explanation' 
    },
    { 
      window: 'Window 2', 
      timeRange: '00:00:48 - 00:01:35',
      complaintLevel: 1,
      complaintPercentage: 25,
      description: 'Initial frustration expressed'
    },
    { 
      window: 'Window 3', 
      timeRange: '00:01:48 - 00:02:42',
      complaintLevel: 2,
      complaintPercentage: 100,
      description: '"Product clearly defective"'
    },
    { 
      window: 'Window 4', 
      timeRange: '00:02:57 - 00:03:54',
      complaintLevel: 1,
      complaintPercentage: 50,
      description: '"Wasted too much time"'
    },
    { 
      window: 'Window 5', 
      timeRange: '00:04:21 - 00:05:09',
      complaintLevel: 1,
      complaintPercentage: 25,
      description: '"Should work out of the box"'
    },
    { 
      window: 'Window 6', 
      timeRange: '00:05:22 - 00:06:09',
      complaintLevel: 1,
      complaintPercentage: 50,
      description: '"Still not happy"'
    },
    { 
      window: 'Window 7', 
      timeRange: '00:06:24 - 00:07:07',
      complaintLevel: 0,
      complaintPercentage: 0,
      description: 'Progress toward resolution'
    },
    { 
      window: 'Window 8', 
      timeRange: '00:07:28 - 00:08:22',
      complaintLevel: 1,
      complaintPercentage: 25,
      description: '"Still frustrating"'
    },
    { 
      window: 'Window 9', 
      timeRange: '00:08:31 - 00:09:16',
      complaintLevel: 0,
      complaintPercentage: 0,
      description: 'Positive conclusion'
    }
  ];

  // Calculate the average complaint percentage
  const totalPercentage = data.reduce((sum, item) => sum + item.complaintPercentage, 0);
  const averagePercentage = (totalPercentage / data.length).toFixed(1);

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto p-6 rounded-lg shadow-2xl border border-green-500" style={{
      background: `linear-gradient(135deg, ${colors.dark.base} 0%, ${colors.dark.secondary} 100%)`,
      boxShadow: `0 0 20px ${colors.neon.darkGreen}`
    }}>
      <h2 className="text-2xl font-bold mb-6 font-mono uppercase tracking-widest" style={{color: colors.neon.green}}>TERMINAL // CUSTOMER COMPLAINT MATRIX</h2>
      
      <div className="w-full mb-8 border border-green-400 rounded-lg p-4" style={{
        background: 'rgba(0, 20, 0, 0.7)', 
        boxShadow: `0 0 15px ${colors.neon.darkGreen}`
      }}>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.1)" />
            <XAxis 
              dataKey="window" 
              angle={-45} 
              textAnchor="end" 
              height={70} 
              tick={{ fontSize: 12, fill: colors.text.primary, fontFamily: 'monospace' }}
              stroke={colors.text.secondary}
            />
            <YAxis 
              yAxisId="left" 
              label={{ value: 'COMPLAINT LEVEL', angle: -90, position: 'insideLeft', fill: colors.neon.green, fontSize: 12, fontFamily: 'monospace' }} 
              tick={{ fill: colors.text.primary, fontFamily: 'monospace' }}
              stroke={colors.text.secondary}
            />
            <YAxis 
              yAxisId="right" 
              orientation="right" 
              label={{ value: 'PERCENTAGE %', angle: 90, position: 'insideRight', fill: colors.neon.green, fontSize: 12, fontFamily: 'monospace' }} 
              tick={{ fill: colors.text.primary, fontFamily: 'monospace' }}
              stroke={colors.text.secondary}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: 'rgba(0, 20, 0, 0.9)', 
                borderColor: colors.neon.green,
                color: colors.neon.green,
                boxShadow: `0 0 10px ${colors.neon.green}`,
                borderRadius: '0px',
                fontFamily: 'monospace'
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
                fontFamily: 'monospace'
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
              dot={{ fill: colors.neon.green, r: 4, strokeWidth: 0 }}
              activeDot={{ fill: colors.neon.brightGreen, r: 6, strokeWidth: 0 }}
              animationDuration={1500}
              style={{filter: `drop-shadow(0 0 8px ${colors.neon.green})`}}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="w-full bg-gray-900 p-4 rounded-lg shadow-lg mb-6 border border-green-500" style={{
        background: 'rgba(0, 20, 0, 0.7)', 
        boxShadow: `0 0 15px ${colors.neon.darkGreen}`
      }}>
        <h3 className="text-lg font-mono font-semibold mb-2 text-center uppercase tracking-wider" style={{color: colors.neon.green}}>// SYSTEM DIAGNOSTICS //</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-black p-3 rounded-lg shadow-lg border border-green-500 flex flex-col items-center" style={{boxShadow: `0 0 10px ${colors.neon.darkGreen}`}}>
            <span className="text-sm font-mono text-gray-400 uppercase">CMPLNT.RATIO</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.green}}>{averagePercentage}%</span>
          </div>
          <div className="bg-black p-3 rounded-lg shadow-lg border border-green-500 flex flex-col items-center" style={{boxShadow: `0 0 10px ${colors.neon.darkGreen}`}}>
            <span className="text-sm font-mono text-gray-400 uppercase">PEAK.ALARM</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.green}}>WINDOW_3</span>
          </div>
          <div className="bg-black p-3 rounded-lg shadow-lg border border-green-500 flex flex-col items-center" style={{boxShadow: `0 0 10px ${colors.neon.darkGreen}`}}>
            <span className="text-sm font-mono text-gray-400 uppercase">RUNTIME</span>
            <span className="text-2xl font-bold font-mono" style={{color: colors.neon.green}}>8M:47S</span>
          </div>
        </div>
      </div>
      
      <div className="w-full">
        <h3 className="text-lg font-mono font-semibold mb-2 text-center uppercase tracking-wider" style={{color: colors.neon.green}}>// CONVERSATION DATASTREAM //</h3>
        <div className="overflow-x-auto" style={{boxShadow: `0 0 15px ${colors.neon.darkGreen}`}}>
          <table className="min-w-full bg-black border border-green-500">
            <thead>
              <tr className="bg-gray-900 border-b border-green-500">
                <th className="p-2 border-r border-green-500 font-mono text-xs" style={{color: colors.neon.green}}>SEQUENCE</th>
                <th className="p-2 border-r border-green-500 font-mono text-xs" style={{color: colors.neon.green}}>TIMESTAMP</th>
                <th className="p-2 border-r border-green-500 font-mono text-xs" style={{color: colors.neon.green}}>STATUS</th>
                <th className="p-2 font-mono text-xs" style={{color: colors.neon.green}}>DATA_INPUT</th>
              </tr>
            </thead>
            <tbody style={{fontFamily: 'monospace'}}>
              {data.map((item, index) => (
                <tr key={index} className={`border-b border-green-900 ${index % 2 === 0 ? 'bg-black' : 'bg-gray-900'}`}>
                  <td className="p-2 border-r border-green-900 font-mono font-medium text-xs" style={{color: colors.neon.green}}>{item.window}</td>
                  <td className="p-2 border-r border-green-900 font-mono text-xs" style={{color: colors.neon.darkGreen}}>{item.timeRange}</td>
                  <td className="p-2 border-r border-green-900">
                    <span className={`px-2 py-1 text-xs font-semibold font-mono
                      ${item.complaintLevel === 0 ? 'bg-black text-green-400 border border-green-500' : 
                        item.complaintLevel === 1 ? 'bg-black text-yellow-400 border border-yellow-500' : 
                        'bg-black text-red-400 border border-red-500'}`}
                      style={{
                        boxShadow: item.complaintLevel === 0 ? `0 0 5px ${colors.neon.green}` : 
                                  item.complaintLevel === 1 ? `0 0 5px ${colors.neon.yellow}` : 
                                  `0 0 5px red`
                      }}>
                      {item.complaintLevel === 0 ? 'STABLE' : 
                       item.complaintLevel === 1 ? 'CAUTION' : 'ALERT'}
                    </span>
                  </td>
                  <td className="p-2 font-mono text-xs" style={{color: colors.neon.darkGreen}}>{item.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="w-full text-center mt-4 font-mono text-xs" style={{color: colors.neon.darkGreen}}>
        <p>CORP//MAINFRAME ACCESS [v3.5.2] © 2025 [RESTRICTED]</p>
        <p>SESSION: XR-5529 | SCAN COMPLETE | TERMINAL: ACTIVE</p>
      </div>
    </div>
  );
};

export default ComplaintTrackingChart;