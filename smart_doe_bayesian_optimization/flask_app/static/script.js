document.addEventListener('DOMContentLoaded', function() {
    const modelButton = document.getElementById('initiateModelButton');
    const optimizerButton = document.getElementById('initiateOptimizerButton');
    const inputButton = document.getElementById('performOptimizationIteration');
    const statusMessage = document.getElementById('statusMessage');
    const modelImage = document.getElementById('modelImage');
    const acqFuncImage = document.getElementById('acqFuncImage');
    const tableContainer = document.getElementById('tableContainer');
    const userInputField = document.getElementById('userInput');  // Store the input field element

    modelButton.addEventListener('click', function() {
        fetch('/activate_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showStatusMessage(data.message);
                modelImage.src = modelImage.src.split('?')[0] + '?time=' + new Date().getTime();
                generateTable(data.num_parameters);
                console.log('New image src:', modelImage.src);
            } else {
                showStatusMessage(`Error: ${data.message}`);
            }
        })
        .catch(error => {
            showStatusMessage(`Error: ${error.message}`);
        });
    });

    optimizerButton.addEventListener('click', function() {
        fetch('/activate_optimizer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showStatusMessage(data.message);
                acqFuncImage.src = acqFuncImage.src.split('?')[0] + '?time=' + new Date().getTime();
                updateTable(data.next_parameter_setting);
            } else {
                showStatusMessage(`Error: ${data.message}`);
            }
        })
        .catch(error => {
            showStatusMessage(`Error: ${error.message}`);
        });
    });

    inputButton.addEventListener('click', function() {
        const userInput = document.getElementById('userInput').value;
        fetch('/perform_optimization', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ userInput: userInput })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showStatusMessage(data.message);
                acqFuncImage.src = acqFuncImage.src.split('?')[0] + '?time=' + new Date().getTime();
                modelImage.src = modelImage.src.split('?')[0] + '?time=' + new Date().getTime();
                updateTable(data.next_parameter_setting);
                userInputField.value = '';  // Clear the input field
            } else {
                showStatusMessage(`Error: ${data.message}`);
            }
        })
        .catch(error => {
            showStatusMessage(`Error: ${error.message}`);
        });
    });

    function generateTable(numParameters) {
        tableContainer.innerHTML = ''; // Clear any existing table

        const table = document.createElement('table');

        const headerRow = document.createElement('tr');
        const headers = ['Parameter', 'Next Setting'];
        headers.forEach(headerText => {
            const header = document.createElement('th');
            header.textContent = headerText;
            headerRow.appendChild(header);
        });
        table.appendChild(headerRow);

        for (let i = 0; i < numParameters; i++) {
            const row = document.createElement('tr');

            const parameterCell = document.createElement('td');
            parameterCell.textContent = `Parameter ${i + 1}`;
            row.appendChild(parameterCell);

            const recommendationCell = document.createElement('td');
            recommendationCell.textContent = `-`; // Placeholder text
            recommendationCell.id = `recommendation-${i + 1}`; // Set ID for the cell
            row.appendChild(recommendationCell);

            table.appendChild(row);
        }

        tableContainer.appendChild(table);
    }

    function updateTable(nextParameterSetting) {
        for (let i = 0; i < nextParameterSetting.length; i++) {
            const recommendationCell = document.getElementById(`recommendation-${i + 1}`);
            if (recommendationCell) {
                const value = Number(nextParameterSetting[i]);  // Convert to number
                const formattedValue = value.toFixed(4);  // Format to 4 decimal places
                recommendationCell.textContent = formattedValue;
            }
        }
    }

    function showStatusMessage(message) {
        const statusMessageOverlay = document.getElementById('statusMessageOverlay');
        statusMessageOverlay.textContent = message;
        statusMessageOverlay.classList.add('show');
    
        // Hide the overlay after 3 seconds
        setTimeout(function() {
            statusMessageOverlay.classList.remove('show');
        }, 1000);
    }
    
});
