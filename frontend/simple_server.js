// Serveur de test simple pour vérifier que React fonctionne
const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Servir les fichiers statiques
app.use(express.static(path.join(__dirname, 'public')));

// Route principale
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>M&A Intelligence Platform</title>
        <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
        <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    </head>
    <body>
        <div id="root"></div>
        <script type="text/babel">
            function App() {
                return React.createElement('div', {
                    style: { 
                        padding: '20px', 
                        fontFamily: 'Arial, sans-serif',
                        textAlign: 'center'
                    }
                }, [
                    React.createElement('h1', { key: 'title' }, 'M&A Intelligence Platform'),
                    React.createElement('p', { key: 'subtitle' }, 'Backend et Frontend en cours de démarrage...'),
                    React.createElement('div', { 
                        key: 'status',
                        style: { 
                            backgroundColor: '#10b981', 
                            color: 'white', 
                            padding: '10px', 
                            borderRadius: '5px',
                            marginTop: '20px'
                        }
                    }, '✅ Frontend de test fonctionnel')
                ]);
            }
            
            ReactDOM.render(React.createElement(App), document.getElementById('root'));
        </script>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`🚀 M&A Intelligence Platform test server running on http://localhost:${PORT}`);
});