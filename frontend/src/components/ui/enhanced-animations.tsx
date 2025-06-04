/**
 * Animations et Micro-interactions Haut de Gamme - M&A Intelligence Platform
 * Sprint 6 - Système d'animations sophistiqué pour UX premium
 */

import React, { ReactNode, useState, useEffect } from 'react';
import { motion, AnimatePresence, useInView, useMotionValue, useSpring } from 'framer-motion';
import { cn } from '../../lib/utils';

// Types pour les animations
export interface AnimationConfig {
  type: 'fadeIn' | 'slideUp' | 'slideDown' | 'slideLeft' | 'slideRight' | 'scale' | 'rotate' | 'bounce' | 'elastic';
  duration?: number;
  delay?: number;
  stagger?: number;
  repeat?: boolean;
  hover?: boolean;
  whileTap?: boolean;
}

// Variantes d'animations prédéfinies
export const animationVariants = {
  fadeIn: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 }
  },
  slideUp: {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  },
  slideDown: {
    hidden: { opacity: 0, y: -20 },
    visible: { opacity: 1, y: 0 }
  },
  slideLeft: {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  },
  slideRight: {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0 }
  },
  scale: {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1 }
  },
  rotate: {
    hidden: { opacity: 0, rotate: -10 },
    visible: { opacity: 1, rotate: 0 }
  },
  bounce: {
    hidden: { opacity: 0, y: -100 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        type: "spring",
        bounce: 0.4,
        duration: 0.8
      }
    }
  },
  elastic: {
    hidden: { opacity: 0, scale: 0 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 10
      }
    }
  }
};

// Composant principal pour animations automatiques
export const AnimatedContainer: React.FC<{
  children: ReactNode;
  config: AnimationConfig;
  className?: string;
  triggerOnView?: boolean;
}> = ({ children, config, className, triggerOnView = true }) => {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  
  const variants = animationVariants[config.type];
  
  const motionConfig = {
    initial: "hidden",
    animate: triggerOnView ? (isInView ? "visible" : "hidden") : "visible",
    variants,
    transition: {
      duration: config.duration || 0.5,
      delay: config.delay || 0,
      ease: "easeOut"
    },
    whileHover: config.hover ? { scale: 1.05, transition: { duration: 0.2 } } : undefined,
    whileTap: config.whileTap ? { scale: 0.95 } : undefined
  };

  return (
    <motion.div
      ref={ref}
      className={className}
      {...motionConfig}
    >
      {children}
    </motion.div>
  );
};

// Composant pour animations en cascade
export const StaggeredContainer: React.FC<{
  children: ReactNode[];
  stagger?: number;
  className?: string;
}> = ({ children, stagger = 0.1, className }) => {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true });

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: stagger,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div
      ref={ref}
      className={className}
      variants={containerVariants}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
    >
      {children.map((child, index) => (
        <motion.div key={index} variants={itemVariants}>
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
};

// Hook pour animations de compteur
export const useCountAnimation = (end: number, duration: number = 2000) => {
  const [count, setCount] = useState(0);
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { duration });

  useEffect(() => {
    const unsubscribe = springValue.on("change", (latest) => {
      setCount(Math.floor(latest));
    });
    
    motionValue.set(end);
    
    return () => unsubscribe();
  }, [end, motionValue, springValue]);

  return count;
};

// Composant de compteur animé
export const AnimatedCounter: React.FC<{
  end: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}> = ({ end, duration = 2000, prefix = "", suffix = "", className }) => {
  const count = useCountAnimation(end, duration);
  
  return (
    <span className={className}>
      {prefix}{count.toLocaleString()}{suffix}
    </span>
  );
};

// Composant de barre de progression animée
export const AnimatedProgress: React.FC<{
  value: number;
  max?: number;
  className?: string;
  showValue?: boolean;
  color?: string;
  duration?: number;
}> = ({ value, max = 100, className, showValue = false, color = "primary", duration = 1000 }) => {
  const [animatedValue, setAnimatedValue] = useState(0);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [value]);

  const percentage = Math.min((animatedValue / max) * 100, 100);

  return (
    <div className={cn("relative w-full bg-muted rounded-full h-2", className)}>
      <motion.div
        className={cn("h-2 rounded-full", `bg-${color}`)}
        initial={{ width: "0%" }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: duration / 1000, ease: "easeOut" }}
      />
      {showValue && (
        <motion.span
          className="absolute right-0 top-3 text-xs text-muted-foreground"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: duration / 2000 }}
        >
          {Math.round(percentage)}%
        </motion.span>
      )}
    </div>
  );
};

// Composant de notification toast animée
export const AnimatedToast: React.FC<{
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  isVisible: boolean;
  onClose: () => void;
  duration?: number;
}> = ({ message, type = 'info', isVisible, onClose, duration = 3000 }) => {
  
  const getToastStyles = () => {
    const baseStyles = "px-4 py-2 rounded-lg shadow-lg border";
    switch (type) {
      case 'success': return cn(baseStyles, "bg-green-50 border-green-200 text-green-800");
      case 'error': return cn(baseStyles, "bg-red-50 border-red-200 text-red-800");
      case 'warning': return cn(baseStyles, "bg-yellow-50 border-yellow-200 text-yellow-800");
      default: return cn(baseStyles, "bg-blue-50 border-blue-200 text-blue-800");
    }
  };

  useEffect(() => {
    if (isVisible && duration > 0) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [isVisible, duration, onClose]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -50, scale: 0.9 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className={cn("fixed top-4 right-4 z-50", getToastStyles())}
        >
          <div className="flex items-center justify-between gap-2">
            <span>{message}</span>
            <button
              onClick={onClose}
              className="ml-2 text-current opacity-50 hover:opacity-100 transition-opacity"
            >
              ✕
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Composant de skeleton loading animé
export const AnimatedSkeleton: React.FC<{
  className?: string;
  lines?: number;
  width?: string | string[];
}> = ({ className, lines = 1, width = "100%" }) => {
  const widths = Array.isArray(width) ? width : Array(lines).fill(width);

  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, index) => (
        <motion.div
          key={index}
          className="h-4 bg-muted rounded"
          style={{ width: widths[index] || "100%" }}
          animate={{
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  );
};

// Composant de particules animées
export const ParticleAnimation: React.FC<{
  count?: number;
  className?: string;
  color?: string;
}> = ({ count = 20, className, color = "#3B82F6" }) => {
  const particles = Array.from({ length: count }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 4 + 1,
    duration: Math.random() * 3 + 2
  }));

  return (
    <div className={cn("absolute inset-0 overflow-hidden pointer-events-none", className)}>
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full opacity-20"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: particle.size,
            height: particle.size,
            backgroundColor: color
          }}
          animate={{
            y: [0, -100, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  );
};

// Hook pour gestion des micro-interactions
export const useMicroInteractions = () => {
  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const handlers = {
    onMouseEnter: () => setIsHovered(true),
    onMouseLeave: () => setIsHovered(false),
    onMouseDown: () => setIsPressed(true),
    onMouseUp: () => setIsPressed(false),
    onFocus: () => setIsFocused(true),
    onBlur: () => setIsFocused(false),
  };

  const getAnimationProps = () => ({
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.98 },
    whileFocus: { boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.5)" },
    transition: { type: "spring", stiffness: 300, damping: 30 }
  });

  return {
    handlers,
    states: { isHovered, isPressed, isFocused },
    getAnimationProps
  };
};

// Composant bouton avec micro-interactions
export const InteractiveButton: React.FC<{
  children: ReactNode;
  className?: string;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  disabled?: boolean;
}> = ({ children, className, variant = 'primary', size = 'md', onClick, disabled = false }) => {
  const { handlers, getAnimationProps } = useMicroInteractions();

  const getVariantStyles = () => {
    switch (variant) {
      case 'primary': return "bg-primary text-primary-foreground shadow-md";
      case 'secondary': return "bg-secondary text-secondary-foreground border";
      case 'ghost': return "hover:bg-muted hover:text-muted-foreground";
      default: return "bg-primary text-primary-foreground";
    }
  };

  const getSizeStyles = () => {
    switch (size) {
      case 'sm': return "px-3 py-1.5 text-sm";
      case 'lg': return "px-6 py-3 text-lg";
      default: return "px-4 py-2";
    }
  };

  return (
    <motion.button
      className={cn(
        "rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2",
        getVariantStyles(),
        getSizeStyles(),
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
      onClick={onClick}
      disabled={disabled}
      {...handlers}
      {...getAnimationProps()}
    >
      {children}
    </motion.button>
  );
};

export default {
  AnimatedContainer,
  StaggeredContainer,
  AnimatedCounter,
  AnimatedProgress,
  AnimatedToast,
  AnimatedSkeleton,
  ParticleAnimation,
  InteractiveButton,
  useCountAnimation,
  useMicroInteractions
};