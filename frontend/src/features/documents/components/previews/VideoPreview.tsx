/**
 * Composant Preview Vidéo - M&A Intelligence Platform
 * Sprint 3 - Lecteur vidéo simple
 */

import React, { useState, useRef, useCallback } from 'react';
import { Button } from '../../../../components/ui/button';
import { cn } from '../../../../lib/utils';
import { Document, PreviewConfig } from '../../types';

interface VideoPreviewProps {
  document: Document;
  config: PreviewConfig;
}

const VideoPreview: React.FC<VideoPreviewProps> = ({ document, config }) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [volume, setVolume] = useState<number>(1);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showControls, setShowControls] = useState<boolean>(true);
  const [hasError, setHasError] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const controlsTimeoutRef = useRef<NodeJS.Timeout>();

  // Gestion lecture/pause
  const togglePlayPause = useCallback(() => {
    if (!videoRef.current) return;

    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  // Gestion du temps
  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return;
    setCurrentTime(videoRef.current.currentTime);
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    if (!videoRef.current) return;
    setDuration(videoRef.current.duration);
    setHasError(false);
  }, []);

  const handleSeek = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!videoRef.current) return;
    const time = parseFloat(event.target.value);
    videoRef.current.currentTime = time;
    setCurrentTime(time);
  }, []);

  // Gestion volume
  const handleVolumeChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!videoRef.current) return;
    const newVolume = parseFloat(event.target.value);
    videoRef.current.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  }, []);

  const toggleMute = useCallback(() => {
    if (!videoRef.current) return;
    
    if (isMuted) {
      videoRef.current.volume = volume;
      videoRef.current.muted = false;
      setIsMuted(false);
    } else {
      videoRef.current.muted = true;
      setIsMuted(true);
    }
  }, [isMuted, volume]);

  // Plein écran
  const toggleFullscreen = useCallback(() => {
    if (!videoRef.current) return;

    if (!isFullscreen) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }, [isFullscreen]);

  // Gestion des contrôles
  const showControlsTemporarily = useCallback(() => {
    setShowControls(true);
    
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current);
    }
    
    controlsTimeoutRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false);
      }
    }, 3000);
  }, [isPlaying]);

  const handleMouseMove = useCallback(() => {
    showControlsTemporarily();
  }, [showControlsTemporarily]);

  // Formatage du temps
  const formatTime = useCallback((time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  // Gestion erreurs
  const handleError = useCallback(() => {
    setHasError(true);
  }, []);

  // Events listeners pour fullscreen
  React.useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Auto-play configuration
  React.useEffect(() => {
    if (config.autoPlay && videoRef.current && !hasError) {
      videoRef.current.play().catch(() => {
        // Auto-play bloqué par le navigateur
        setIsPlaying(false);
      });
    }
  }, [config.autoPlay, hasError]);

  if (hasError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-ma-red-500">
        <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-center mb-2">Erreur de chargement de la vidéo</p>
        <p className="text-sm text-center text-ma-slate-500 mb-4">
          Format non supporté ou fichier corrompu
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => window.open(document.downloadUrl, '_blank')}
        >
          Télécharger la vidéo
        </Button>
      </div>
    );
  }

  return (
    <div 
      className="relative bg-black h-full flex items-center justify-center"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => isPlaying && setShowControls(false)}
    >
      {/* Vidéo */}
      <video
        ref={videoRef}
        src={document.url}
        className="max-w-full max-h-full"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onError={handleError}
        onClick={togglePlayPause}
        poster={document.thumbnailUrl}
      />

      {/* Overlay de contrôles */}
      <div className={cn(
        "absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent transition-opacity duration-300",
        showControls ? "opacity-100" : "opacity-0"
      )}>
        {/* Bouton play central */}
        {!isPlaying && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Button
              variant="ghost"
              size="icon"
              className="h-16 w-16 rounded-full bg-white/20 hover:bg-white/30 backdrop-blur-sm"
              onClick={togglePlayPause}
            >
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </Button>
          </div>
        )}

        {/* Contrôles du bas */}
        <div className="absolute bottom-0 left-0 right-0 p-4">
          {/* Barre de progression */}
          <div className="mb-3">
            <input
              type="range"
              min={0}
              max={duration || 0}
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-1 bg-white/30 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentTime / duration) * 100}%, rgba(255,255,255,0.3) ${(currentTime / duration) * 100}%, rgba(255,255,255,0.3) 100%)`
              }}
            />
          </div>

          {/* Contrôles */}
          <div className="flex items-center justify-between text-white">
            {/* Gauche - Play/Pause et temps */}
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white hover:bg-white/20"
                onClick={togglePlayPause}
              >
                {isPlaying ? (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                )}
              </Button>

              <span className="text-sm font-mono">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>

            {/* Droite - Volume et plein écran */}
            <div className="flex items-center gap-3">
              {/* Volume */}
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white hover:bg-white/20"
                  onClick={toggleMute}
                >
                  {isMuted || volume === 0 ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                  )}
                </Button>

                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  className="w-20 h-1 bg-white/30 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Plein écran */}
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white hover:bg-white/20"
                onClick={toggleFullscreen}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Informations vidéo */}
      {duration > 0 && (
        <div className="absolute top-4 left-4 bg-black/75 text-white text-xs px-2 py-1 rounded">
          {Math.round(document.size / (1024 * 1024))} MB • {formatTime(duration)}
        </div>
      )}

      {/* Instructions */}
      {showControls && (
        <div className="absolute top-4 right-4 bg-black/75 text-white text-xs px-2 py-1 rounded">
          Cliquez pour jouer/pause
        </div>
      )}
    </div>
  );
};

export default VideoPreview;