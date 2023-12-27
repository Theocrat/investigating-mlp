function makeColor(intensity, maxIntensity) {
    // Value = 0: 250, 200, 100
    // Value Max: 250, 250, 250
    // Value Min: 100,   0,  50

    if (intensity == 0) {
        return `250, 200, 100`
    }

    if (intensity > maxIntensity) {
        return `250, 250, 250`
    }

    if (intensity < -maxIntensity) {
        return `100, 0, 50`
    }

    if (intensity > 0) {
        let green = 200 + 50 * (intensity / maxIntensity)
        let blue = 100 + 150 * (intensity / maxIntensity)
        return `250, ${green}, ${blue}`
    }

    if (intensity < 0) {
        let red = 250 + 150 * (intensity / maxIntensity)
        let green = 200 + 200 * (intensity / maxIntensity)
        let blue = 100 + 50 * (intensity / maxIntensity)
        return `${red}, ${green}, ${blue}`
    }

    // The following code should not be reachable, but in case modification
    // to the code above causes control flow to leak through, the greyish 
    // blue color here should stand out as an indication
    return `150, 150, 220`
}

function setLayer(targetSVG, dataSource, eventFunc, maxIntensityInput) {
    let epoch = parseInt(frame.value)
    let ncols = dataSource[epoch].ncols
    let nrows = dataSource[epoch].nrows
    let maxIntensity = document.getElementById(maxIntensityInput).value

    let heatmapBlocks = [`
    <rect x="0" y="0" width="${ncols}" height="${nrows}"
          style="fill:#4d6;" />
    `]
    for (let row = 0; row < nrows; row++) {
        for (let col = 0; col < ncols; col++) {
            let colorCode = makeColor(
                dataSource[epoch].data[row][col], 
                maxIntensity
            )
            heatmapBlocks.push(`
            <rect x="${col}" y="${row}" width="1" height="1"
                  style="fill:rgb(${colorCode});" class="heat-block"
                  onmouseover="${eventFunc}(${epoch}, ${row}, ${col})" />
            `)
        }
    }

    targetSVG.innerHTML = heatmapBlocks.join("\n")
}

function hoverStatsWeight(epoch, row, col) {
    let value = Math.round(1000 * modelWeightsL1[epoch].data[row][col]) / 1000
    document.getElementById("hover-weight").innerHTML = `
    <span class="key">row</span>: <span class="value">${row}</span><br/>
    <span class="key">col</span>: <span class="value">${col}</span><br/>
    <span class="key">value</span>: <span class="value">${value}</span>
    `
}

function hoverStatsGrad(epoch, row, col) {
    let value = Math.round(1000 * modelGradsL1[epoch].data[row][col]) / 1000
    document.getElementById("hover-grad").innerHTML = `
    <span class="key">row</span>: <span class="value">${row}</span><br/>
    <span class="key">col</span>: <span class="value">${col}</span><br/>
    <span class="key">value</span>: <span class="value">${value}</span>
    `
}