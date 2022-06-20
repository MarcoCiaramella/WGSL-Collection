
async function getGPUDevice() {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        alert("No adapter");
    }
    else {
        return await adapter.requestDevice();
    }
}

getGPUDevice()
    .then(async device => {

        async function sha256(bytesArray) {

            const messages = [];
            let bufferSize = 0;
            const sizes = getMessageSizes(bytesArray[0]);
            bytesArray.forEach(bytes => {
                if (bytes.length % 4 !== 0) throw "Message must be 32-bit aligned";
                const message = padMessage(bytes, sizes[1]);
                // message is the padded version of the input message as dscribed by SHA-256 specification
                messages.push(message);
                // messages has same size
                bufferSize += message.byteLength;
            });
            const numMessages = messages.length;

            // build shader input data
            const messageArray = new Uint32Array(new ArrayBuffer(bufferSize));
            let offset = 0;
            messages.forEach(message => {
                messageArray.set(message, offset);
                offset += message.length;
            });

            // messages
            const messageArrayBuffer = device.createBuffer({
                mappedAtCreation: true,
                size: messageArray.byteLength,
                usage: GPUBufferUsage.STORAGE
            });
            new Uint32Array(messageArrayBuffer.getMappedRange()).set(messageArray);
            messageArrayBuffer.unmap();

            // sizes
            const sizesBuffer = device.createBuffer({
                mappedAtCreation: true,
                size: sizes.byteLength,
                usage: GPUBufferUsage.STORAGE
            });
            new Uint32Array(sizesBuffer.getMappedRange()).set(sizes);
            sizesBuffer.unmap();

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
                            buffer: sizesBuffer
                        }
                    },
                    {
                        binding: 2,
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
            passEncoder.dispatchWorkgroups(1, 1);
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
            for (let i = 0; i < numMessages; i++) {
                console.log(buf2hex(gpuReadBuffer.getMappedRange(i * hashSize, hashSize)));
            }
        }

        // at the current version of WGSL u64 is not supported. This force the max message length to be ((2^32) - 1) / 32
        const messages = [
            [0x01, 0x00, 0x00, 0x00], // int 1
            [0x02, 0x00, 0x00, 0x00], // int 2
            [0x03, 0x00, 0x00, 0x00], // int 3
            [0x04, 0x00, 0x00, 0x00], // int 4
            [0x05, 0x00, 0x00, 0x00], // int 5
            [0x06, 0x00, 0x00, 0x00], // int 6
            [0x07, 0x00, 0x00, 0x00], // int 7
            [0x08, 0x00, 0x00, 0x00], // int 8
            [0x09, 0x00, 0x00, 0x00]  // int 9
        ];
        // each message in messages must have the same size
        await sha256(messages);
    })
    .catch(err => console.error(err));

function buf2hex(buffer) {
    return new Uint8Array(buffer).reduce((a, b) => a + b.toString(16).padStart(2, '0'), '0x');
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
