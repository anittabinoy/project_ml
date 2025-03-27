$(document).ready(function() {
    // Simple Linear Regression
    $('#simple-linear-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_type: 'simple_linear',
                area: $('#simple-area').val()
            }),
            success: function(response) {
                $('#simple-linear-result').text(response.prediction);
            }
        });
    });

    // Multiple Linear Regression
    $('#multiple-linear-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_type: 'multiple_linear',
                area: $('#multiple-area').val(),
                bedrooms: $('#bedrooms').val(),
                bathrooms: $('#bathrooms').val(),
                distance: $('#distance').val(),
                parking: $('#parking').val()
            }),
            success: function(response) {
                $('#multiple-linear-result').text(response.prediction);
            }
        });
    });

    // Polynomial Regression
    $('#polynomial-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_type: 'polynomial',
                hour: $('#hour').val()
            }),
            success: function(response) {
                $('#polynomial-result').text(response.prediction);
            }
        });
    });

    // KNN Regression
    $('#knn-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_type: 'knn',
                study_hours: $('#study-hours').val(),
                sleep_hours: $('#sleep-hours').val()
            }),
            success: function(response) {
                $('#knn-result').text(response.prediction);
            }
        });
    });

    // Logistic Regression
    $('#logistic-form').on('submit', function(e) {
        e.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_type: 'logistic',
                exam1_score: $('#exam1-score').val(),
                exam2_score: $('#exam2-score').val()
            }),
            success: function(response) {
                $('#logistic-result').text(response.prediction);
                $('#logistic-probability').text(response.probability);
            }
        });
    });
});
