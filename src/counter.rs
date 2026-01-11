use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{hex_digit1, multispace0, multispace1},
    combinator::{map, map_res, opt},
    multi::separated_list1,
    sequence::{delimited, preceded, separated_pair},
};
use std::str;

#[derive(Debug, PartialEq)]
pub struct Counter {
    pub name: String,
    pub desc: String,
    pub event: u64,
    pub umask: u64,
}

impl Counter {
    pub fn parse(i: &[u8]) -> Result<Self, String> {
        match Self::parse_nom(i) {
            Ok((_, counter)) => Ok(counter),
            Err(e) => Err(format!("Parse error: {:?}", e)),
        }
    }

    pub fn parse_nom(i: &[u8]) -> IResult<&[u8], Counter> {
        let (i, _) = multispace0(i)?;
        let (i, name) = parse_name(i)?;
        let (i, _) = multispace1(i)?;
        let (i, desc) = map(delimited(tag("["), take_until("]\n"), tag("]")), |s| {
            str::from_utf8(s)
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .parse(i)?;
        let (i, _) = multispace1(i)?;
        let (i, (event, umask)) = parse_config(i)?;
        let (i, _) = multispace0(i)?;

        Ok((
            i,
            Counter {
                name,
                desc,
                event,
                umask,
            },
        ))
    }
}

fn parse_hex(i: &[u8]) -> IResult<&[u8], u64> {
    map_res(preceded(opt(tag("0x")), hex_digit1), |out: &[u8]| {
        u64::from_str_radix(str::from_utf8(out).unwrap(), 16)
    })
    .parse(i)
}

fn parse_key_value(i: &[u8]) -> IResult<&[u8], (&str, u64)> {
    separated_pair(
        map(alt((tag("event"), tag("umask"))), |s| {
            str::from_utf8(s).unwrap()
        }),
        tag("="),
        parse_hex,
    )
    .parse(i)
}

fn parse_config(i: &[u8]) -> IResult<&[u8], (u64, u64)> {
    let (i, _) = tag("cpu/")(i)?;
    let (i, kvs) = separated_list1(tag(","), parse_key_value).parse(i)?;
    let (i, _) = tag("/")(i)?;

    let mut event = 0;
    let mut umask = 0;
    for (k, v) in kvs {
        match k {
            "event" => event = v,
            "umask" => umask = v,
            _ => {}
        }
    }
    Ok((i, (event, umask)))
}

fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-'
}

fn parse_name(i: &[u8]) -> IResult<&[u8], String> {
    map(take_while1(is_name_char), |s| {
        str::from_utf8(s).unwrap().to_owned()
    })
    .parse(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bp_l1_btb_correct() {
        let text = br#"
  bp_l1_btb_correct                                 
       [L1 Branch Prediction Overrides Existing Prediction (speculative)]
        cpu/event=0x8a/
"#;

        let res = Counter::parse(text).unwrap();

        assert_eq!(
            res,
            Counter {
                name: "bp_l1_btb_correct".to_owned(),
                desc: "L1 Branch Prediction Overrides Existing Prediction (speculative)".to_owned(),
                event: 0x8a,
                umask: 0,
            }
        );
    }
    #[test]
    fn bp_l1_tlb_fetch_hit() {
        let text = br#"
  bp_l1_tlb_fetch_hit                               
       [The number of instruction fetches that hit in the L1 ITLB]
        cpu/umask=0xff,event=0x94/ 
"#;

        let res = Counter::parse(text).unwrap();

        assert_eq!(
            res,
            Counter {
                name: "bp_l1_tlb_fetch_hit".to_owned(),
                desc: "The number of instruction fetches that hit in the L1 ITLB".to_owned(),
                event: 0x94,
                umask: 0xff,
            }
        );
    }
    #[test]
    fn fp_ret_sse_avx_ops_all() {
        let text = br#"
  fp_ret_sse_avx_ops.all
       [All FLOPS. This is a retire-based event. The number of retired SSE/AVX
        FLOPS. The number of events logged per cycle can vary from 0 to 64.
        This event can count above 15]
        cpu/umask=0xff,event=0x3/
"#;

        let res = Counter::parse(text).unwrap();

        assert_eq!(
            res,
            Counter {
                name: "fp_ret_sse_avx_ops.all".to_owned(),
                desc: "All FLOPS. This is a retire-based event. The number of retired SSE/AVX FLOPS. The number of events logged per cycle can vary from 0 to 64. This event can count above 15"
                    .to_owned(),
                event: 0x3,
                umask: 0xff,
            }
        );
    }

    #[test]
    fn ex_ret_mmx_fp_instr_sse_instr() {
        let text = br#"
  ex_ret_mmx_fp_instr.sse_instr
       [SSE instructions (SSE, SSE2, SSE3, SSSE3, SSE4A, SSE41, SSE42, AVX)]
        cpu/umask=0x4,event=0xcb/
"#;

        let res = Counter::parse(text).unwrap();

        assert_eq!(
            res,
            Counter {
                name: "ex_ret_mmx_fp_instr.sse_instr".to_owned(),
                desc: "SSE instructions (SSE, SSE2, SSE3, SSSE3, SSE4A, SSE41, SSE42, AVX)"
                    .to_owned(),
                event: 0xcb,
                umask: 0x4,
            }
        );
    }

    #[test]
    fn ex_tagged_ibs_ops_ibs_count_rollover() {
        let text = br#"
  ex_tagged_ibs_ops.ibs_count_rollover       
       [Tagged IBS Ops. Number of times an op could not be tagged by IBS
        because of a previous tagged op that has not retired]
        cpu/umask=0x4,event=0x1cf/"#;
        let res = Counter::parse(text).unwrap();
        assert_eq!(res.name, "ex_tagged_ibs_ops.ibs_count_rollover");
    }
}
