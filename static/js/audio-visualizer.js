// Audio Visualizer for generated speech
document.addEventListener('DOMContentLoaded', function() {
    // Get the audio element
    const audioElement = document.getElementById('audio-player');
    if (!audioElement) return;

    // Get the canvas element
    const canvas = document.getElementById('visualizer');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Create audio context
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    let source = null;
    let animationId = null;
    let isPlaying = false;
    
    // Set canvas dimensions
    function resizeCanvas() {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
    }
    
    // Resize canvas initially and on window resize
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Initialize visualizer when play is clicked
    function initializeVisualizer() {
        if (!audioContext) {
            // Create new audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create analyser node
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            
            // Create buffer for frequency data
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            // Connect audio source to analyser
            source = audioContext.createMediaElementSource(audioElement);
            source.connect(analyser);
            analyser.connect(audioContext.destination);
        }
    }
    
    // Draw the visualization
    function draw() {
        if (!isPlaying) return;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Get frequency data
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate bar width based on canvas width and data array length
        const barWidth = canvas.width / dataArray.length * 2.5;
        let barHeight;
        let x = 0;
        
        // Set gradients for bars
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, 'rgba(0, 210, 255, 0.8)');
        gradient.addColorStop(1, 'rgba(0, 100, 150, 0.2)');
        
        // Draw bars
        for (let i = 0; i < dataArray.length; i++) {
            barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
            
            // Skip some bars for aesthetic reasons
            if (i % 2 === 0) {
                ctx.fillStyle = gradient;
                
                // Draw rounded bars
                ctx.beginPath();
                // Use regular rect for browsers that don't support roundRect
                if (ctx.roundRect) {
                    ctx.roundRect(
                        x, 
                        canvas.height - barHeight, 
                        barWidth - 1, 
                        barHeight, 
                        [3, 3, 0, 0]
                    );
                } else {
                    ctx.rect(
                        x, 
                        canvas.height - barHeight, 
                        barWidth - 1, 
                        barHeight
                    );
                }
                ctx.fill();
            }
            
            x += barWidth;
        }
        
        // Draw circular visualization in the center
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(canvas.width, canvas.height) * 0.2;
        
        const circleGradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, radius
        );
        circleGradient.addColorStop(0, 'rgba(0, 210, 255, 0.7)');
        circleGradient.addColorStop(1, 'rgba(0, 100, 150, 0)');
        
        // Get average amplitude for circle size
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const average = sum / dataArray.length;
        
        // Draw pulse circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * (0.5 + average / 255 * 0.5), 0, Math.PI * 2);
        ctx.fillStyle = circleGradient;
        ctx.fill();
        
        // Continue animation loop
        animationId = requestAnimationFrame(draw);
    }
    
    // Handle audio play event
    audioElement.addEventListener('play', function() {
        isPlaying = true;
        
        // Initialize audio context if not already done
        if (audioContext === null) {
            initializeVisualizer();
        } else {
            // Resume audio context if it was suspended
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        
        // Start animation
        if (!animationId) {
            animationId = requestAnimationFrame(draw);
        }
    });
    
    // Handle audio pause event
    audioElement.addEventListener('pause', function() {
        isPlaying = false;
        
        // Stop animation
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
    
    // Handle audio ended event
    audioElement.addEventListener('ended', function() {
        isPlaying = false;
        
        // Stop animation
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
});