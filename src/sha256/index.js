async function getGPUDevice() {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        throw "No adapter";
    }
    else {
        return await adapter.requestDevice();
    }
}

function padMessage(bytes, size) {
    const arrBuff = new ArrayBuffer(size * 4);
    new Uint8Array(arrBuff).set(bytes);
    return new Uint32Array(arrBuff);
}

function getMessageSizes(bytes) {
    const lenBit = bytes.length * 8;
    const k = 512 - (lenBit + 1 + 64) % 512;
    const padding = 1 + k + 64;
    const lenBitPadded = lenBit + padding;
    const arrBuff = new ArrayBuffer(2 * Uint32Array.BYTES_PER_ELEMENT);
    const u32Arr = new Uint32Array(arrBuff);
    u32Arr[0] = lenBit / 32;
    u32Arr[1] = lenBitPadded / 32;
    return u32Arr;
}

function calcNumWorkgroups(device, messages) {
    const numWorkgroups = Math.ceil(messages.length / 256);
    if (numWorkgroups > device.limits.maxComputeWorkgroupsPerDimension) {
        throw `Input array too large. Max size is ${device.limits.maxComputeWorkgroupsPerDimension / 256}.`;
    }
    return numWorkgroups;
}

let device;

/**
 * 
 * @param {Uint8Array[]} messages messages to hash. Each message must be 32-bit aligned with the same size
 * @returns {Uint8Array[]} hashes
 */
async function sha256(messages) {

    for (const message of messages) {
        if (message.length !== messages[0].length) throw "Messages must have the same size";
    }

    device = device ? device : await getGPUDevice();

    const numWorkgroups = calcNumWorkgroups(device, messages);

    const messagesPad = [];
    let bufferSize = 0;
    const messageSizes = getMessageSizes(messages[0]);
    for (const message of messages) {
        if (message.length % 4 !== 0) throw "Message must be 32-bit aligned";
        const messagePad = padMessage(message, messageSizes[1]);
        // message is the padded version of the input message as dscribed by SHA-256 specification
        messagesPad.push(messagePad);
        // messages has same size
        bufferSize += messagePad.byteLength;
    }
    const numMessages = messagesPad.length;

    // build shader input data
    const messageArray = new Uint32Array(new ArrayBuffer(bufferSize));
    let offset = 0;
    for (const message of messagesPad) {
        messageArray.set(message, offset);
        offset += message.length;
    }

    // messages
    const messageArrayBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: messageArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(messageArrayBuffer.getMappedRange()).set(messageArray);
    messageArrayBuffer.unmap();

    // num_messages
    const numMessagesBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(numMessagesBuffer.getMappedRange()).set([messagesPad.length]);
    numMessagesBuffer.unmap();

    // message_sizes
    const messageSizesBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: messageSizes.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(messageSizesBuffer.getMappedRange()).set(messageSizes);
    messageSizesBuffer.unmap();

    // Result
    const resultBufferSize = (256 / 8) * numMessages;
    const resultBuffer = device.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const shaderModule = device.createShaderModule({
        code: await (await fetch("shader.wgsl")).text()
    });

    const computePipeline = device.createComputePipeline({
        compute: {
            module: shaderModule,
            entryPoint: "sha256"
        },
        layout: 'auto'
    });

    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: messageArrayBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: numMessagesBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: messageSizesBuffer
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: resultBuffer
                }
            }
        ]
    });

    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const gpuReadBuffer = device.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(
        resultBuffer,
        0,
        gpuReadBuffer,
        0,
        resultBufferSize
    );

    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);

    const hashSize = 256 / 8;
    const hashes = [];
    for (let i = 0; i < numMessages; i++) {
        hashes.push(new Uint8Array(gpuReadBuffer.getMappedRange(i * hashSize, hashSize)));
    }

    return hashes;
}

// at the current version of WGSL u64 is not supported. This force the max message length to be ((2^32) - 1) / 32
const messages = [
    new Uint8Array([0x01, 0x00, 0x00, 0x00]), // int 1
    new Uint8Array([0x02, 0x00, 0x00, 0x00]), // int 2
    new Uint8Array([0x03, 0x00, 0x00, 0x00]), // int 3
    new Uint8Array([0x04, 0x00, 0x00, 0x00]), // int 4
    new Uint8Array([0x05, 0x00, 0x00, 0x00]), // int 5
    new Uint8Array([0x06, 0x00, 0x00, 0x00]), // int 6
    new Uint8Array([0x07, 0x00, 0x00, 0x00]), // int 7
    new Uint8Array([0x08, 0x00, 0x00, 0x00]), // int 8
    new Uint8Array([0x09, 0x00, 0x00, 0x00])  // int 9
];
// each message in messages must have the same size
sha256(messages)
    .then(hashes => {
        for (const hash of hashes) {
            console.log(hash.reduce((a, b) => a + b.toString(16).padStart(2, '0'), ''));
        }
    })
    .catch(err => console.error(err));