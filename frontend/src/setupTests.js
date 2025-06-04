<!-- 
  Créez ce fichier temporaire favicon-generator.html et ouvrez-le dans un navigateur 
  pour générer les images nécessaires. Faites clic droit > Enregistrer l'image sous...
-->
<!DOCTYPE html>
<html>
<head>
    <title>Favicon Generator</title>
</head>
<body style="background: #f0f0f0; padding: 20px;">
    <h2>M&A Intelligence Platform - Favicons</h2>
    <p>Faites clic droit sur chaque image et "Enregistrer l'image sous..." dans le dossier public/</p>
    
    <!-- Favicon 16x16 -->
    <div style="margin: 20px 0;">
        <h3>favicon.ico (16x16)</h3>
        <canvas id="favicon16" width="16" height="16" style="border: 1px solid #ccc; background: white;"></canvas>
    </div>
    
    <!-- Logo 192x192 -->
    <div style="margin: 20px 0;">
        <h3>logo192.png</h3>
        <canvas id="logo192" width="192" height="192" style="border: 1px solid #ccc; background: white;"></canvas>
    </div>
    
    <!-- Logo 512x512 -->
    <div style="margin: 20px 0;">
        <h3>logo512.png</h3>
        <canvas id="logo512" width="512" height="512" style="border: 1px solid #ccc; background: white;"></canvas>
    </div>

    <script>
        // Fonction pour dessiner le logo M&A
        function drawLogo(canvasId, size) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            // Background
            ctx.fillStyle = '#1976d2';
            ctx.fillRect(0, 0, size, size);
            
            // Text "M&A" 
            ctx.fillStyle = 'white';
            ctx.font = `bold ${size * 0.35}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('M&A', size / 2, size / 2.2);
            
            // Underline
            ctx.strokeStyle = 'white';
            ctx.lineWidth = size * 0.05;
            ctx.beginPath();
            ctx.moveTo(size * 0.2, size * 0.65);
            ctx.lineTo(size * 0.8, size * 0.65);
            ctx.stroke();
            
            // Small text "Intelligence"
            if (size > 100) {
                ctx.font = `${size * 0.08}px Arial`;
                ctx.fillText('INTELLIGENCE', size / 2, size * 0.8);
            }
        }
        
        // Générer les logos
        drawLogo('favicon16', 16);
        drawLogo('logo192', 192);
        drawLogo('logo512', 512);
        
        // Pour le favicon.ico, on doit le convertir
        document.getElementById('favicon16').toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'favicon.png';
            document.body.appendChild(link);
        });
    </script>
</body>
</html>