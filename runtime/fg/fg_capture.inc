// A capture of what the library is given, from a running game, to try it again offline (fg_harness.cpp reads it).
// Included by nvapi_fg.cpp. Making fg_capture.trigger beside the DLL makes the next four evaluations write their inputs:
// cap_<frame>_<n>.bin, each a header of width, height, DXGI format and row size (four 32 bit numbers) and the rows.

namespace {

bool read_texture(ID3D12CommandQueue *queue, ID3D12Resource *texture, uint32_t state, std::vector<uint8_t> &rows,
                  uint32_t &width, uint32_t &height, uint32_t &format, uint32_t &row_size) {
    ComPtr<ID3D12Device> device;
    queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = total;
    buffer_desc.Height = buffer_desc.DepthOrArraySize = buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(readback.GetAddressOf()))) ||
        FAILED(device->CreateCommandAllocator(queue->GetDesc().Type, IID_PPV_ARGS(allocator.GetAddressOf()))) ||
        FAILED(device->CreateCommandList(0, queue->GetDesc().Type, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()))))
        return false;

    dlss5nr::transition(list.Get(), texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION destination{}, source{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    source.pResource = texture;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    dlss5nr::transition(list.Get(), texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    list->Close();
    ID3D12CommandList *lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), 1);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, 5000);
    CloseHandle(event);

    void *mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    width = (uint32_t)desc.Width;
    height = desc.Height;
    format = (uint32_t)desc.Format;
    const uint32_t texel = format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R8G8B8A8_UNORM ||
                                   format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_R16G16_FLOAT
                               ? 4
                               : 4;
    row_size = width * texel;
    rows.resize((size_t)row_size * height);
    for (uint32_t y = 0; y < height; ++y)
        memcpy(&rows[(size_t)y * row_size], static_cast<const uint8_t *>(mapped) + (size_t)y * footprint.Footprint.RowPitch, row_size);
    readback->Unmap(0, nullptr);
    return true;
}

// The inputs of an evaluation, in the order the passes are in, when a capture is asked for.
void capture_inputs(ID3D12CommandQueue *queue, const std::vector<Pass> &passes) {
    static int remaining = 0, frame = 0;
    if (remaining == 0) {
        const std::wstring trigger = g_dir + L"fg_capture.trigger";
        if (GetFileAttributesW(trigger.c_str()) == INVALID_FILE_ATTRIBUTES) return;
        DeleteFileW(trigger.c_str());
        remaining = 4;
        frame = 0;
        say("capture: four evaluations");
    }

    int n = 0;
    for (const Pass &p : passes) {
        std::vector<uint8_t> rows;
        uint32_t width, height, format, row_size;
        if (!read_texture(queue, p.mirror.Get(), D3D12_RESOURCE_STATE_COMMON, rows, width, height, format, row_size)) {
            say("capture: a texture could not be read");
            continue;
        }
        wchar_t name[64];
        swprintf(name, 64, L"cap_%d_%d.bin", frame, n);
        FILE *f = _wfopen((g_dir + name).c_str(), L"wb");
        if (f) {
            const uint32_t header[4] = {width, height, format, row_size};
            fwrite(header, 4, 4, f);
            fwrite(rows.data(), 1, rows.size(), f);
            fclose(f);
        }
        ++n;
    }
    say("capture: frame %d, %d textures", frame, n);
    ++frame;
    --remaining;
}

} // namespace
