//! Picks the PulseAudio monitor source to capture system audio from.
//!
//! Pure selection logic for [`super::linux`], compiled on every target
//! so its tests run on all CI runners.

/// An input source as the host lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Source {
    /// Backend identifier. On the PulseAudio host this is the source
    /// name, `<sink>.monitor` for monitors.
    pub id: String,
    /// Human-readable description, e.g. `Monitor of Built-in Audio`.
    pub name: String,
}

impl Source {
    /// Whether the source mirrors an output sink.
    pub(crate) fn is_monitor(&self) -> bool {
        let name = self.name.to_lowercase();
        self.id.ends_with(".monitor") || name.contains("monitor of") || name.contains("monitor_of")
    }
}

/// Index of the monitor to capture: the `requested` one (matched by id
/// or name), else the monitor of `default_sink`, else the first monitor.
/// `None` when no source is a monitor.
pub(crate) fn choose_monitor(
    sources: &[Source],
    requested: Option<&str>,
    default_sink: Option<&str>,
) -> Option<usize> {
    let monitors = || sources.iter().enumerate().filter(|(_, s)| s.is_monitor());
    let by = |pick: &dyn Fn(&Source) -> bool| monitors().find(|(_, s)| pick(s)).map(|(i, _)| i);

    requested
        .and_then(|want| by(&|s| s.id == want || s.name == want))
        .or_else(|| {
            let monitor_id = format!("{}.monitor", default_sink?);
            by(&|s| s.id == monitor_id)
        })
        .or_else(|| monitors().next().map(|(i, _)| i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn source(id: &str, name: &str) -> Source {
        Source {
            id: id.into(),
            name: name.into(),
        }
    }

    fn pulse_sources() -> Vec<Source> {
        vec![
            source(
                "alsa_input.pci-0000_00_1f.3.analog-stereo",
                "Built-in Audio Analog Stereo",
            ),
            source(
                "alsa_output.pci-0000_00_1f.3.hdmi-stereo.monitor",
                "Monitor of Built-in Audio Digital Stereo (HDMI)",
            ),
            source(
                "alsa_output.usb-headset.analog-stereo.monitor",
                "Monitor of USB Headset Analog Stereo",
            ),
        ]
    }

    #[test]
    fn monitors_are_told_apart_from_microphones() {
        let flags: Vec<bool> = pulse_sources().iter().map(Source::is_monitor).collect();
        assert_eq!(flags, vec![false, true, true]);
    }

    #[test]
    fn alsa_bridge_names_still_count_as_monitors() {
        assert!(source("hw:1", "Monitor of Built-in Audio").is_monitor());
        assert!(source("pulse", "monitor_of_sink").is_monitor());
        assert!(!source("pulse", "PulseAudio Sound Server").is_monitor());
    }

    #[test]
    fn the_default_sink_monitor_wins_over_the_first_monitor() {
        let got = choose_monitor(
            &pulse_sources(),
            None,
            Some("alsa_output.usb-headset.analog-stereo"),
        );
        assert_eq!(got, Some(2));
    }

    #[test]
    fn a_requested_monitor_wins_by_id_or_name() {
        let sources = pulse_sources();
        let sink = Some("alsa_output.usb-headset.analog-stereo");
        let by_id = choose_monitor(
            &sources,
            Some("alsa_output.pci-0000_00_1f.3.hdmi-stereo.monitor"),
            sink,
        );
        let by_name = choose_monitor(
            &sources,
            Some("Monitor of Built-in Audio Digital Stereo (HDMI)"),
            sink,
        );
        assert_eq!((by_id, by_name), (Some(1), Some(1)));
    }

    #[test]
    fn a_requested_microphone_is_ignored() {
        let got = choose_monitor(
            &pulse_sources(),
            Some("alsa_input.pci-0000_00_1f.3.analog-stereo"),
            None,
        );
        assert_eq!(got, Some(1));
    }

    #[test]
    fn an_unknown_default_sink_falls_back_to_the_first_monitor() {
        assert_eq!(
            choose_monitor(&pulse_sources(), None, Some("default")),
            Some(1)
        );
        assert_eq!(
            choose_monitor(&pulse_sources(), Some("gone"), None),
            Some(1)
        );
    }

    #[test]
    fn no_monitor_means_no_choice() {
        let mics = vec![source("alsa_input.usb", "USB Mic")];
        assert_eq!(choose_monitor(&mics, None, Some("alsa_output.usb")), None);
    }
}
