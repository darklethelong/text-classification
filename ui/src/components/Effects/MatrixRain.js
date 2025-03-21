import React, { useEffect, useRef } from 'react';

const MatrixRain = ({ 
  opacity = 0.15,
  speed = 1,
  density = 1,
  fontSize = 16,
  characterSet = 'matrix' // 'matrix', 'binary', 'custom'
}) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Make canvas fill the screen
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // Get character set based on option
    let chars = '';
    switch(characterSet) {
      case 'binary':
        chars = '01';
        break;
      case 'custom':
        chars = 'COMPLAINTALGORITHM0123456789';
        break;
      case 'matrix':
      default:
        chars = 'ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,./<>?';
    }
    
    const columns = Math.floor(canvas.width / fontSize);
    
    // Arrays to store the current y position and speed of each column
    const drops = [];
    const dropSpeed = [];
    
    for (let i = 0; i < columns; i++) {
      drops[i] = Math.random() * -100; // Start above the canvas
      dropSpeed[i] = Math.random() * 0.5 + 0.5; // Random speed between 0.5 and 1
    }
    
    // Matrix rain drawing function
    const draw = () => {
      // Translucent black background to create fade effect
      ctx.fillStyle = `rgba(0, 0, 0, 0.1)`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw characters
      for (let i = 0; i < drops.length; i++) {
        // Skip some columns for a more natural look
        if (Math.random() > density) continue;
        
        // Random character
        const text = chars[Math.floor(Math.random() * chars.length)];
        
        // Vary the brightness for a more dynamic effect
        const brightness = Math.floor(Math.random() * 55) + 200; // Range: 200-255 (increased from 100-255)
        ctx.fillStyle = `rgba(0, ${brightness}, 100, 1.0)`; // Increased green intensity
        ctx.font = `${fontSize}px monospace`;
        
        // Add text shadow for a glow effect
        ctx.shadowBlur = 5;
        ctx.shadowColor = "rgba(0, 255, 0, 0.7)";
        
        // x = i * fontSize, y = drops[i] * fontSize
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        // Reset shadow to avoid performance issues
        ctx.shadowBlur = 0;
        
        // Extra bright 'head' character
        if (Math.random() > 0.95) {
          ctx.fillStyle = `rgba(200, 255, 200, 1)`;
          ctx.shadowBlur = 10;
          ctx.shadowColor = "rgba(0, 255, 0, 0.9)";
          ctx.fillText(text, i * fontSize, drops[i] * fontSize);
          ctx.shadowBlur = 0;
        }
        
        // Move drops down at their individual speed
        drops[i] += dropSpeed[i] * speed;
        
        // Randomize drop speeds
        if (Math.random() > 0.95) {
          dropSpeed[i] = Math.random() * 0.5 + 0.5;
        }
        
        // Send it back to the top randomly after it reaches the bottom
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
          // Reset speed
          dropSpeed[i] = Math.random() * 0.5 + 0.5;
        }
      }
    };
    
    // Animation loop
    let animationId;
    const animate = () => {
      draw();
      animationId = requestAnimationFrame(animate);
    };
    
    animate();
    
    // Clean up
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, [opacity, speed, density, fontSize, characterSet]);
  
  return (
    <canvas 
      ref={canvasRef}
      className="matrix-effect"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        opacity: opacity,
        pointerEvents: 'none'
      }}
    />
  );
};

export default MatrixRain; 