import { forwardRef, type ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const BASE =
  "inline-flex items-center justify-center gap-1.5 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 focus-visible:ring-offset-1 disabled:pointer-events-none disabled:opacity-50";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-accent-600 text-white shadow-sm hover:bg-accent-700 active:scale-[0.98]",
  secondary:
    "border border-subtle bg-surface-elevated text-content-secondary hover:bg-surface-sunken active:scale-[0.98]",
  ghost:
    "text-content-tertiary hover:text-content-secondary hover:bg-surface-sunken",
  danger:
    "bg-rose-600 text-white shadow-sm hover:bg-rose-700 active:scale-[0.98]",
};

const SIZE: Record<Size, string> = {
  sm: "rounded-md px-2.5 py-1 text-ui-xs",
  md: "rounded-lg px-3.5 py-1.5 text-ui-sm",
  lg: "rounded-full px-6 py-2.5 text-ui-md",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button({ variant = "secondary", size = "md", className, ...rest }, ref) {
    return (
      <button
        ref={ref}
        type="button"
        className={`${BASE} ${VARIANT[variant]} ${SIZE[size]}${className ? ` ${className}` : ""}`}
        {...rest}
      />
    );
  },
);
