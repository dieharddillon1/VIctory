//! End-to-end tests for heap-indexed 64-bit image atomics.

use super::host::{Dispatch, Host, Root, View, supports_typed_image_atomics64};
use super::{HostShader, validator::Validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, FourCC, dxbc, io, ir, resource, structure};

fn wide_atomics() -> HostShader {
    let words = [
        0xbe81_0380u32, // s_mov_b32 s1, 0
        0xf40c_0100,
        0xfa00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0
        0x7e00_0280, // v_mov_b32 v0, 0 (texel x)
        0x7e02_0280, // v_mov_b32 v1, 0 (texel y)
        0x7e04_02c1, // v_mov_b32 v2, -1
        0x7e06_0281, // v_mov_b32 v3, 1
        0xf03c_1308,
        0x0001_0200, // image_atomic_swap: texel 0 = 0x1ffffffff
        0x7e04_0282, // v_mov_b32 v2, 2
        0x7e06_0283, // v_mov_b32 v3, 3
        0xf044_3308,
        0x0001_0200, // image_atomic_add glc: carry into the high dword
        0xbf8c_0000, // s_waitcnt 0: wait for GLC return
        0x7e00_0281, // v_mov_b32 v0, 1
        0xf03c_1308,
        0x0001_0200, // texel 1 = value before add
        0x7e00_0280, // v_mov_b32 v0, 0
        0x7e04_0287, // v_mov_b32 v2, 7 (replacement)
        0x7e06_0289, // v_mov_b32 v3, 9
        0x7e08_0281, // v_mov_b32 v4, 1 (comparand)
        0x7e0a_0285, // v_mov_b32 v5, 5
        0xf040_3f08,
        0x0001_0200, // image_atomic_cmpswap glc
        0xbf8c_0000, // s_waitcnt 0
        0x7e00_0282, // v_mov_b32 v0, 2
        0xf03c_1308,
        0x0001_0200, // texel 2 = value before compare-exchange
        0x7e00_0280, // v_mov_b32 v0, 0
        0x7e04_0289, // v_mov_b32 v2, 9
        0x7e06_02ff,
        0x8000_0000, // v_mov_b32 v3, literal (unsigned max)
        0xf05c_3308,
        0x0001_0200, // image_atomic_umax glc
        0xbf8c_0000, // s_waitcnt 0
        0x7e00_0283, // v_mov_b32 v0, 3
        0xf03c_1308,
        0x0001_0200, // texel 3 = value before unsigned max
        0xbf81_0000, // s_endpgm
    ];
    shader_from_words(&words)
}

fn narrow_atomics() -> HostShader {
    shader_from_words(&[
        0xbe81_0380, // s_mov_b32 s1, 0
        0xf40c_0100,
        0xfa00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0
        0x7e00_0280, // v_mov_b32 v0, 0 (texel x)
        0x7e02_0280, // v_mov_b32 v1, 0 (texel y)
        0x7e04_0287, // v_mov_b32 v2, 7
        0xf03c_1108,
        0x0001_0200, // image_atomic_swap: texel 0 = 7
        0x7e04_0282, // v_mov_b32 v2, 2
        0xf044_3108,
        0x0001_0200, // image_atomic_add glc
        0xbf8c_0000, // s_waitcnt 0
        0x7e00_0281, // v_mov_b32 v0, 1
        0xf03c_1108,
        0x0001_0200, // image_atomic_swap: texel 1 = value before add
        0xbf81_0000, // s_endpgm
    ])
}

fn shader_from_words(words: &[u32]) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let program = ir::build(&native.code, false, native.vgprs).unwrap();
    let structured = structure::structure(&program).unwrap();
    let io = io::raise(&program, &native).unwrap();
    let res = resource::track(&program, &native.layout, HwStage::Cs).unwrap();
    let bytes = dxbc::build(&[]);
    let container = Container::parse(&bytes).unwrap();
    super::emit(&container, &native, &program, &structured, &io, &res).unwrap()
}

#[test]
fn typed_heap_image_atomics_validate_the_feature_flags() {
    let mut shader = wide_atomics();
    assert!(shader.needs_typed_image_atomics64());
    let container = Container::parse(&shader.dxil).unwrap();
    let info = container.part(FourCC::SFI0).unwrap().data;
    let features = u64::from_le_bytes(info.try_into().unwrap());
    for expected in [0x8000, 0x40_0000, 0x1000_0000] {
        assert_ne!(features & expected, 0);
    }
    if let Some(path) = Validator::default_path() {
        Validator::load(Some(&path))
            .unwrap()
            .validate(&mut shader.dxil)
            .unwrap();
    }
}

#[test]
fn typed_heap_image_atomics_read_and_write_both_dwords() {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    let requested = std::env::var("VICTORY_ADAPTER").ok();
    let (host, label) = if let Some(wanted) = requested {
        // SAFETY: plain DXGI enumeration and D3D12 creation.
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().unwrap();
            let wanted = wanted.to_lowercase();
            let (adapter, name) = (0..)
                .map_while(|i| factory.EnumAdapters1(i).ok())
                .find_map(|adapter| {
                    let name = adapter.GetDesc1().ok()?.Description;
                    let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
                    let name = String::from_utf16_lossy(&name[..end]);
                    name.to_lowercase()
                        .contains(&wanted)
                        .then_some((adapter, name))
                })
                .expect("requested hardware adapter");
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
            (
                Host {
                    device: device.unwrap(),
                },
                name,
            )
        }
    } else {
        let Ok(host) = Host::new(false) else { return };
        (host, "default adapter".into())
    };
    let narrow = narrow_atomics();
    assert!(!narrow.needs_typed_image_atomics64());
    let (root, pipeline) = host.compute_pipeline(&narrow).unwrap();
    let narrow_readback = host
        .dispatch(
            &root,
            &pipeline,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; narrow.user_data_dwords() as usize]),
                )],
                views: &[View::UintTextureUav(2, 1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    assert_eq!(narrow_readback[0], [9, 7], "{label}: 32-bit regression");
    if !supports_typed_image_atomics64(&host.device) {
        eprintln!("{label}: typed descriptor-heap image atomics are unavailable");
        return;
    }
    let shader = wide_atomics();
    let (root, pipeline) = host.compute_pipeline(&shader).unwrap();
    let readback = host
        .dispatch(
            &root,
            &pipeline,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                )],
                views: &[View::Uint64TextureUav(4, 1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    assert_eq!(
        readback[0],
        [9, 0x8000_0000, 0xffff_ffff, 1, 1, 5, 7, 9],
        "{label}"
    );
    eprintln!(
        "{label}: image swap, add carry, compare-exchange and unsigned max GPU readback matched"
    );
}
