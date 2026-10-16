//! Interleaving of Core Media audio buffer lists.
//!
//! An `AudioBufferList` carries either one interleaved buffer holding
//! every channel, or one planar buffer per channel. ScreenCaptureKit
//! delivers the planar form, so concatenating the buffers (as the
//! system-audio handler did before) produced `[L L … R R …]` labelled as
//! interleaved stereo: a left-only 440 Hz tone came out as 880 Hz on
//! both channels with silent halves. This module turns either layout
//! into the interleaved frames the rest of the pipeline expects.
//!
//! Pure and platform-independent so the behaviour is unit-tested on
//! every OS, not only where ScreenCaptureKit exists.

/// One buffer of a list: its declared channel count and its samples.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plane<'a> {
    /// `number_channels` of the buffer: 1 for a planar buffer, the
    /// full count for an interleaved one.
    pub channels: u32,
    /// Float32 samples of the buffer.
    pub samples: &'a [f32],
}

/// Interleaved frames with `expected_channels` channels built from the
/// buffers of one `AudioBufferList`, and whether the list's layout was
/// one of the two well-formed ones.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Interleaved {
    pub samples: Vec<f32>,
    /// `false` when the layout did not match `expected_channels` and
    /// the samples were remapped (channels duplicated or dropped).
    pub layout_matched: bool,
}

/// Build interleaved frames from the buffers of one list.
///
/// - One buffer with `expected_channels` channels: passthrough.
/// - `expected_channels` buffers of one channel each: interleaved.
/// - Any other layout: every buffer is treated as a plane (an
///   interleaved buffer with more channels contributes its first
///   channel), missing channels repeat the last available plane and
///   extra planes are dropped, so timing is preserved and
///   `layout_matched` is `false`.
pub(crate) fn interleave(buffers: &[Plane<'_>], expected_channels: u16) -> Interleaved {
    let expected = usize::from(expected_channels.max(1));
    match buffers {
        [] => Interleaved {
            samples: Vec::new(),
            layout_matched: true,
        },
        [single] if single.channels as usize == expected => Interleaved {
            samples: single.samples.to_vec(),
            layout_matched: true,
        },
        planes => {
            let layout_matched = planes.len() == expected && planes.iter().all(|p| p.channels == 1);
            // Frames available in a plane: its samples per declared channel.
            let frames = planes
                .iter()
                .map(|p| p.samples.len() / p.channels.max(1) as usize)
                .min()
                .unwrap_or(0);
            let mut samples = Vec::with_capacity(frames * expected);
            for frame in 0..frames {
                for channel in 0..expected {
                    // Missing planes repeat the last one; extra planes are dropped.
                    let plane = &planes[channel.min(planes.len() - 1)];
                    let stride = plane.channels.max(1) as usize;
                    samples.push(plane.samples[frame * stride]);
                }
            }
            Interleaved {
                samples,
                layout_matched,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn plane(channels: u32, samples: &[f32]) -> Plane<'_> {
        Plane { channels, samples }
    }

    #[test]
    fn planar_stereo_is_interleaved_frame_by_frame() {
        let left = [1.0, 2.0, 3.0];
        let right = [-1.0, -2.0, -3.0];
        let out = interleave(&[plane(1, &left), plane(1, &right)], 2);
        assert_eq!(out.samples, vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
        assert!(out.layout_matched);
    }

    #[test]
    fn interleaved_stereo_passes_through() {
        let data = [1.0, -1.0, 2.0, -2.0];
        let out = interleave(&[plane(2, &data)], 2);
        assert_eq!(out.samples, data.to_vec());
        assert!(out.layout_matched);
    }

    #[test]
    fn mono_passes_through() {
        let data = [0.5, 0.25];
        let out = interleave(&[plane(1, &data)], 1);
        assert_eq!(out.samples, data.to_vec());
        assert!(out.layout_matched);
    }

    #[test]
    fn a_single_mono_buffer_is_duplicated_when_stereo_was_expected() {
        let out = interleave(&[plane(1, &[0.5, 0.25])], 2);
        assert_eq!(out.samples, vec![0.5, 0.5, 0.25, 0.25]);
        assert!(!out.layout_matched);
    }

    #[test]
    fn unequal_planes_are_truncated_to_the_shortest() {
        let out = interleave(&[plane(1, &[1.0, 2.0, 3.0]), plane(1, &[-1.0, -2.0])], 2);
        assert_eq!(out.samples, vec![1.0, -1.0, 2.0, -2.0]);
        assert!(out.layout_matched);
    }

    #[test]
    fn extra_planes_are_dropped_when_fewer_channels_were_expected() {
        let out = interleave(&[plane(1, &[1.0]), plane(1, &[2.0]), plane(1, &[3.0])], 2);
        assert_eq!(out.samples, vec![1.0, 2.0]);
        assert!(!out.layout_matched);
    }

    #[test]
    fn an_interleaved_buffer_with_more_channels_keeps_its_first_channel() {
        // One 4-channel interleaved buffer, mono expected.
        let out = interleave(&[plane(4, &[1.0, 9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0])], 1);
        assert_eq!(out.samples, vec![1.0, 2.0]);
        assert!(!out.layout_matched);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        let out = interleave(&[], 2);
        assert!(out.samples.is_empty());
        let out = interleave(&[plane(1, &[]), plane(1, &[])], 2);
        assert!(out.samples.is_empty());
    }
}
