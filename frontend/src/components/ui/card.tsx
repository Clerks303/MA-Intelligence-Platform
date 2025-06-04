import * as React from "react"

import { cn } from "../../lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border bg-card text-card-foreground shadow-sm",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
))
CardFooter.displayName = "CardFooter"

// M&A Intelligence specific card variants
const StatsCard = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    title: string
    value: string | number
    change?: string
    trend?: 'up' | 'down' | 'neutral'
    icon?: React.ReactNode
  }
>(({ className, title, value, change, trend, icon, ...props }, ref) => (
  <Card ref={ref} className={cn("p-6", className)} {...props}>
    <div className="flex items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium text-muted-foreground">
        {title}
      </CardTitle>
      {icon && <div className="h-4 w-4 text-muted-foreground">{icon}</div>}
    </div>
    <div>
      <div className="text-2xl font-bold">{value}</div>
      {change && (
        <p className={cn(
          "text-xs",
          trend === 'up' ? "text-ma-green-600" : 
          trend === 'down' ? "text-ma-red-600" : "text-muted-foreground"
        )}>
          {change}
        </p>
      )}
    </div>
  </Card>
))
StatsCard.displayName = "StatsCard"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent, StatsCard }