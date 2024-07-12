// This code was written by an AI assistant
let imageData;
let originalImageData;

function loadImage(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            context.drawImage(img, 0, 0);
            imageData = context.getImageData(0, 0, img.width, img.height);
            originalImageData = context.getImageData(0, 0, img.width, img.height);
        }
        img.src = e.target.result;
    }
    reader.readAsDataURL(file);
}

function downloadImage() {
    const link = document.createElement('a');
    link.download = 'dithered_image.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function resetImage() {
    if (originalImageData) {
        canvas.width = originalImageData.width;
        canvas.height = originalImageData.height;
        context.putImageData(originalImageData, 0, 0);
        imageData = context.createImageData(originalImageData.width, originalImageData.height);
        imageData.data.set(new Uint8ClampedArray(originalImageData.data));
    }
}

function closestColor(pixel, palette) {
    let minDistance = Infinity;
    let closest = null;
    for (let i = 0; i < palette.length; i++) {
        const distance = Math.sqrt(
            Math.pow(pixel[0] - palette[i][0], 2) +
            Math.pow(pixel[1] - palette[i][1], 2) +
            Math.pow(pixel[2] - palette[i][2], 2)
        );
        if (distance < minDistance) {
            minDistance = distance;
            closest = palette[i];
        }
    }
    return closest;
}

function getBayerMatrix(level) {
    const bayer2 = [
        [0, 2],
        [3, 1]
    ];

    const bayer4 = [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ];

    const bayer8 = [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ];

    const bayer16 = [
        [0, 128, 32, 160, 8, 136, 40, 168, 2, 130, 34, 162, 10, 138, 42, 170],
        [192, 64, 224, 96, 200, 72, 232, 104, 194, 66, 226, 98, 202, 74, 234, 106],
        [48, 176, 16, 144, 56, 184, 24, 152, 50, 178, 18, 146, 58, 186, 26, 154],
        [240, 112, 208, 80, 248, 120, 216, 88, 242, 114, 210, 82, 250, 122, 218, 90],
        [12, 140, 44, 172, 4, 132, 36, 164, 14, 142, 46, 174, 6, 134, 38, 166],
        [204, 76, 236, 108, 196, 68, 228, 100, 206, 78, 238, 110, 198, 70, 230, 102],
        [60, 188, 28, 156, 52, 180, 20, 148, 62, 190, 30, 158, 54, 182, 22, 150],
        [252, 124, 220, 92, 244, 116, 212, 84, 254, 126, 222, 94, 246, 118, 214, 86],
        [3, 131, 35, 163, 11, 139, 43, 171, 1, 129, 33, 161, 9, 137, 41, 169],
        [195, 67, 227, 99, 203, 75, 235, 107, 193, 65, 225, 97, 201, 73, 233, 105],
        [51, 179, 19, 147, 59, 187, 27, 155, 49, 177, 17, 145, 57, 185, 25, 153],
        [243, 115, 211, 83, 251, 123, 219, 91, 241, 113, 209, 81, 249, 121, 217, 89],
        [15, 143, 47, 175, 7, 135, 39, 167, 13, 141, 45, 173, 5, 133, 37, 165],
        [207, 79, 239, 111, 199, 71, 231, 103, 205, 77, 237, 109, 197, 69, 229, 101],
        [63, 191, 31, 159, 55, 183, 23, 151, 61, 189, 29, 157, 53, 181, 21, 149],
        [255, 127, 223, 95, 247, 119, 215, 87, 253, 125, 221, 93, 245, 117, 213, 85]
    ];

    const matrices = [bayer2, bayer4, bayer8, bayer16];
    return matrices[level].map(row => row.map(v => v / (matrices[level].length * matrices[level].length) - 0.5));
}

function getOrderedMatrix(level) {
    const ordered2 = [
        [1, 3],
        [4, 2]
    ];

    const ordered4 = [
        [1, 9, 3, 11],
        [13, 5, 15, 7],
        [4, 12, 2, 10],
        [16, 8, 14, 6]
    ];

    const ordered8 = [
        [1, 33, 9, 41, 3, 35, 11, 43],
        [49, 17, 57, 25, 51, 19, 59, 27],
        [13, 45, 5, 37, 15, 47, 7, 39],
        [61, 29, 53, 21, 63, 31, 55, 23],
        [4, 36, 12, 44, 2, 34, 10, 42],
        [52, 20, 60, 28, 50, 18, 58, 26],
        [16, 48, 8, 40, 14, 46, 6, 38],
        [64, 32, 56, 24, 62, 30, 54, 22]
    ];

    const ordered16 = [
        [1, 129, 33, 161, 9, 137, 41, 169, 3, 131, 35, 163, 11, 139, 43, 171],
        [193, 65, 225, 97, 201, 73, 233, 105, 195, 67, 227, 99, 203, 75, 235, 107],
        [49, 177, 17, 145, 57, 185, 25, 153, 51, 179, 19, 147, 59, 187, 27, 155],
        [241, 113, 209, 81, 249, 121, 217, 89, 243, 115, 211, 83, 251, 123, 219, 91],
        [13, 141, 45, 173, 5, 133, 37, 165, 15, 143, 47, 175, 7, 135, 39, 167],
        [205, 77, 237, 109, 197, 69, 229, 101, 207, 79, 239, 111, 199, 71, 231, 103],
        [61, 189, 29, 157, 53, 181, 21, 149, 63, 191, 31, 159, 55, 183, 23, 151],
        [253, 125, 221, 93, 245, 117, 213, 85, 255, 127, 223, 95, 247, 119, 215, 87],
        [4, 132, 36, 164, 12, 140, 44, 172, 2, 130, 34, 162, 10, 138, 42, 170],
        [196, 68, 228, 100, 204, 76, 236, 108, 194, 66, 226, 98, 202, 74, 234, 106],
        [52, 180, 20, 148, 60, 188, 28, 156, 50, 178, 18, 146, 58, 186, 26, 154],
        [244, 116, 212, 84, 252, 124, 220, 92, 242, 114, 210, 82, 250, 122, 218, 90],
        [8, 136, 40, 168, 0, 128, 32, 160, 14, 142, 46, 174, 6, 134, 38, 166],
        [200, 72, 232, 104, 192, 64, 224, 96, 206, 78, 238, 110, 198, 70, 230, 102],
        [56, 184, 24, 152, 48, 176, 16, 144, 62, 190, 30, 158, 54, 182, 22, 150],
        [248, 120, 216, 88, 240, 112, 208, 80, 254, 126, 222, 94, 246, 118, 214, 86]
    ];

    const matrices = [ordered2, ordered4, ordered8, ordered16];
    console.log(matrices, level);
    return matrices[level].map(row => row.map(v => v / (matrices[level].length * matrices[level].length + 1) - 0.5));
}

function applyColorQuantization(imageData, palette) {
    const data = imageData.data.slice(); // Copy the data
    for (let i = 0; i < data.length; i += 4) {
        const pixel = [data[i], data[i + 1], data[i + 2]];
        const closest = closestColor(pixel, palette);
        data[i] = closest[0];
        data[i + 1] = closest[1];
        data[i + 2] = closest[2];
    }
    return new ImageData(data, imageData.width, imageData.height);
}

function applyBrightness(imageData, brightness) {
    const data = imageData.data.slice(); // Copy the data
    for (let i = 0; i < data.length; i += 4) {
        // Modify the R, G, B values
        data[i] = Math.min(255, Math.max(0, data[i] * brightness));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] * brightness));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] * brightness));
    }
    return new ImageData(data, imageData.width, imageData.height);
}

function applyDithering(imageData, spread, redColorCount, greenColorCount, blueColorCount, pattern, level) {
    let ditherMatrix;
    if (pattern === 'bayer') {
        ditherMatrix = getBayerMatrix(level);
    } else if (pattern === 'ordered') {
        ditherMatrix = getOrderedMatrix(level);
    } else if (pattern === 'threshold') {
        return applyThresholdDithering(imageData, 128);
    } else {
        throw new Error("Unsupported pattern. Choose 'bayer', 'ordered', or 'threshold'.");
    }

    const data = new Uint8ClampedArray(imageData.data);
    const width = imageData.width;
    const height = imageData.height;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const matrixValue = ditherMatrix[y % ditherMatrix.length][x % ditherMatrix[0].length];

            const pixel = [
                data[index],
                data[index + 1],
                data[index + 2]
            ].map(value => value + spread * matrixValue * 255);

            data[index] = Math.floor((redColorCount - 1) * pixel[0] / 255 + 0.5) * (255 / (redColorCount - 1));
            data[index + 1] = Math.floor((greenColorCount - 1) * pixel[1] / 255 + 0.5) * (255 / (greenColorCount - 1));
            data[index + 2] = Math.floor((blueColorCount - 1) * pixel[2] / 255 + 0.5) * (255 / (blueColorCount - 1));
        }
    }

    return new ImageData(data, width, height);
}

function applyThresholdDithering(imageData, threshold) {
    const data = imageData.data.slice(); // Copy the data
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        const value = gray > threshold ? 255 : 0;
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
    }
    return new ImageData(data, imageData.width, imageData.height);
}

function applyMaskToImage(originalImageData, maskImageData, applyWhite) {
    const originalData = originalImageData.data;
    const maskData = maskImageData.data;

    for (let i = 0; i < originalData.length; i += 4) {
        const isBlack = maskData[i] === 0 && maskData[i + 1] === 0 && maskData[i + 2] === 0;
        const isWhite = maskData[i] === 255 && maskData[i + 1] === 255 && maskData[i + 2] === 255;

        if (applyWhite && isWhite) {
            // Apply white mask
            originalData[i] = 255;
            originalData[i + 1] = 255;
            originalData[i + 2] = 255;
        } else if (!applyWhite && isBlack) {
            // Apply black mask
            originalData[i] = 0;
            originalData[i + 1] = 0;
            originalData[i + 2] = 0;
        }
        // Keep the alpha channel unchanged
    }

    return originalImageData;
}

function applyScaling(imageData, scaleMultiplier, resampleFilter) {
    const width = imageData.width;
    const height = imageData.height;
    const newWidth = Math.round(width * 2 ** scaleMultiplier);
    const newHeight = Math.round(height * 2 ** scaleMultiplier);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = newWidth;
    tempCanvas.height = newHeight;
    const tempContext = tempCanvas.getContext('2d');

    if (resampleFilter === 'nearest') {
        tempContext.imageSmoothingEnabled = false;
    } else {
        tempContext.imageSmoothingEnabled = true;
        if (resampleFilter === 'bilinear') {
            tempContext.imageSmoothingQuality = 'medium';
        } else if (resampleFilter === 'bicubic') {
            tempContext.imageSmoothingQuality = 'high';
        }
    }

    tempContext.drawImage(canvas, 0, 0, width, height, 0, 0, newWidth, newHeight);
    return tempContext.getImageData(0, 0, newWidth, newHeight);
}

function ditherImage(pattern, level, imageData) {
    const spread = 0.5;
    const redColorCount = 2;
    const greenColorCount = 2;
    const blueColorCount = 2;

    const ditheredData = applyDithering(imageData, spread, redColorCount, greenColorCount, blueColorCount, pattern, level);
    return ditheredData;
}

function thresholdImage(value, imageData) {
    const ditheredData = applyThresholdDithering(imageData, value);
    return ditheredData;
}

const uploadInput = document.getElementById('upload');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const scaleDownButton = document.getElementById('scaleImageDown');
const scaleUpButton = document.getElementById('scaleImageUp');
const resampleFilterSelect = document.getElementById('resampleFilter');
let resampleFilterState = resampleFilterSelect.value;
const colorQuantizationButton = document.getElementById('applyColorQuantization');
let colorQuantizationState = false;
const thresholdLevelSelect = document.getElementById('thresholdDitherLevel');
let thresholdLevelState = thresholdLevelSelect.value;
const ditherLevelSelect = document.getElementById('ditherLevel');
let ditherLevelState = ditherLevelSelect.value;
const bayerDitherRadio = document.getElementById('bayerDither');
const orderedDitherRadio = document.getElementById('orderedDither');
const noDitherRadio = document.getElementById('noDither');
const clearCanvasButton = document.getElementById('clearCanvas');
const downloadImageButton = document.getElementById('downloadImage');
const thresholdHighlightSelect = document.getElementById('thresholdHighlightLevel');
let thresholdHighlightState = thresholdHighlightSelect.value;
const brightnessSelect = document.getElementById('brightnessLevel');
let brightnessState = brightnessSelect.value;

const applyFilters = () => {
    if (!imageData) return;

    let newImageData = context.createImageData(imageData.width, imageData.height);
    newImageData.data.set(new Uint8ClampedArray(imageData.data)); // Copy the image data

    if (brightnessState && brightnessState != 1) {
        newImageData = applyBrightness(newImageData, brightnessState);
    }

    if (thresholdLevelState && thresholdLevelState != -1) {
        maskData = thresholdImage(thresholdLevelState, newImageData);
        newImageData = applyMaskToImage(newImageData, maskData);
    }

    if (thresholdHighlightState && thresholdHighlightState != -1) {
        maskData = thresholdImage(thresholdHighlightState, newImageData);
        newImageData = applyMaskToImage(newImageData, maskData, true);
    }

    if (!noDitherRadio.checked) {
        console.log('dithering');
        if (bayerDitherRadio.checked) {
            console.log('bayer');
            newImageData = ditherImage('bayer', ditherLevelState, newImageData);
        } else if (orderedDitherRadio.checked) {
            console.log('ordered');
            newImageData = ditherImage('ordered', ditherLevelState, newImageData);
        }
    }


    if (colorQuantizationState) {
        newImageData = applyColorQuantization(newImageData, [
            [255, 16, 16],    // Red
            [0, 134, 49],     // Green
            [0, 56, 206],     // Blue
            [255, 231, 0],    // Yellow
            [0, 0, 0],        // Black
            [255, 255, 255]   // White
        ]);
    }

    context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    canvas.width = newImageData.width;
    canvas.height = newImageData.height;
    context.putImageData(newImageData, 0, 0);
}

downloadImageButton.addEventListener('click', downloadImage);
uploadInput.addEventListener('change', loadImage);
colorQuantizationButton.addEventListener('click', (event) => {
    colorQuantizationState = !colorQuantizationState;
    if (colorQuantizationState) {
        event.target.innerText = 'Remove Color Quantization';
    } else {
        event.target.innerText = 'Apply Color Quantization';
    }
    applyFilters();
});

thresholdLevelSelect.addEventListener('change', (event) => {
    thresholdLevelState = event.target.value;
    document.getElementById('thresholdDitherLevelValue').innerText = thresholdLevelState;
    applyFilters();
});
scaleDownButton.addEventListener('click', (event) => {
    const resampleFilter = resampleFilterSelect.value;
    imageData = applyScaling(imageData, -1, resampleFilter);
    context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    context.putImageData(imageData, 0, 0);
});

scaleUpButton.addEventListener('click', (event) => {
    const resampleFilter = resampleFilterSelect.value;
    imageData = applyScaling(imageData, 1, resampleFilter);
    context.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    context.putImageData(imageData, 0, 0);
});

ditherLevelSelect.addEventListener('change', (event) => {
    ditherLevelState = event.target.value;
    document.getElementById('ditherLevelValue').innerText = ditherLevelState;
    applyFilters();
});

clearCanvasButton.addEventListener('click', resetImage);

resampleFilterSelect.addEventListener('change', (event) => {
    resampleFilterState = event.target.value;
    applyFilters();
});

thresholdHighlightSelect.addEventListener('change', (event) => {
    thresholdHighlightState = 255 - event.target.value;
    document.getElementById('thresholdHighlightLevelValue').innerText = event.target.value;
    applyFilters();
});

bayerDitherRadio.addEventListener('click', (event) => {
    applyFilters();
});

orderedDitherRadio.addEventListener('click', (event) => {
    applyFilters();
});

noDitherRadio.addEventListener('click', (event) => {
    applyFilters();
});

brightnessSelect.addEventListener('change', (event) => {
    brightnessState = event.target.value;
    document.getElementById('brightnessLevelValue').innerText = brightnessState;
    applyFilters();
});