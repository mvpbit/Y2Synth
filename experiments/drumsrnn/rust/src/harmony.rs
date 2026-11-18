//! PAD-led harmony, independent of inference and audio rendering.
use crate::synth1::NoteEvent;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key { pub root: u8, pub minor: bool }
impl Default for Key { fn default() -> Self { Self {root:0,minor:true} } }
impl Key {
    pub fn new(root: u8, minor: bool) -> Self { Self {root:root%12,minor} }
    pub fn contains(self, pitch: u8) -> bool {
        let degree=(pitch%12+12-self.root)%12;
        if self.minor {[0,2,3,5,7,8,10].contains(&degree)}
        else {[0,2,4,5,7,9,11].contains(&degree)}
    }
    pub fn chord_index(self) -> usize {1+self.root as usize+if self.minor {12} else {0}}
    pub fn name(self) -> String {
        format!("{} {}",["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"][self.root as usize],if self.minor {"minor"}else{"major"})
    }
    pub fn snap(self, mut pitch: i32, low: u8) -> u8 {
        while pitch<i32::from(low) {pitch+=12;} while pitch>84 {pitch-=12;}
        (low..=84).filter(|&p|self.contains(p)).min_by_key(|&p|((i32::from(p)-pitch).abs(),p)).unwrap()
    }
}
fn tones(pad: &[NoteEvent], step: usize, key: Key) -> Vec<u8> {
    let mut classes:Vec<u8>=pad.iter().filter(|n|n.step<=step&&step<n.step+n.length_steps&&key.contains(n.pitch)).map(|n|n.pitch%12).collect();
    classes.sort_unstable();classes.dedup();classes
}
pub fn harmonize(notes: &[NoteEvent], pad: &[NoteEvent], key: Key) -> Vec<NoteEvent> {
    let mut out=Vec::new();
    for note in notes {
        let mut start=note.step;
        let stop=(note.step+note.length_steps).min(16);
        while start<stop {
            let chord=tones(pad,start,key);
            let mut end=start+1;
            while end<stop&&tones(pad,end,key)==chord {end+=1;}
            let anchor=!chord.is_empty()&&(start%4==0||end-start>1);
            let mut target=i32::from(note.pitch);
            while target<55 {target+=12;}while target>84 {target-=12;}
            let pitch=(55..=84).filter(|&p|if anchor {chord.contains(&(p%12))}else{key.contains(p)})
                .min_by_key(|&p|((i32::from(p)-target).abs(),p)).unwrap();
            out.push(NoteEvent {step:start,pitch,length_steps:end-start,velocity:note.velocity});
            start=end;
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_keys_keep_scale_and_strong_notes_follow_pad() {
        for root in 0..12 {for minor in [false,true] {
            let key=Key::new(root,minor);
            let pad=vec![NoteEvent {step:0,pitch:key.snap(60,48),length_steps:16,velocity:0.7}];
            let raw=vec![NoteEvent {step:0,pitch:61,length_steps:16,velocity:0.8}];
            let out=harmonize(&raw,&pad,key);
            assert_eq!(out[0].pitch%12,pad[0].pitch%12);
            assert!(out.iter().all(|n|key.contains(n.pitch)));
            assert_eq!(out[0].velocity,0.8);
        }}
    }
    #[test]
    fn held_pluck_changes_pitch_when_pad_harmony_changes() {
        let pad=vec![
            NoteEvent {step:0,pitch:60,length_steps:8,velocity:0.7},
            NoteEvent {step:8,pitch:65,length_steps:8,velocity:0.7},
        ];
        let raw=vec![NoteEvent {step:0,pitch:61,length_steps:16,velocity:0.8}];
        let out=harmonize(&raw,&pad,Key::default());
        assert_eq!(out.len(),2);assert_eq!(out[0].pitch%12,0);assert_eq!(out[1].pitch%12,5);
        assert_eq!(out[0].length_steps,8);assert_eq!(out[1].step,8);
    }
}
