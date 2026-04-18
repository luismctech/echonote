//! Audio capture port.
//!
//! Defines how the application layer drives platform-specific capture
//! engines. Concrete implementations live in `echo-audio` and are compiled
//! conditionally per target OS.
//!
//! The real trait will be shaped in Sprint 0 day 5 together with the Linux
//! and macOS capture adapters.
