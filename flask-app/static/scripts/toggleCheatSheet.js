
// Double Check
$(document).ready(function(){
    console.log("Connected");
});

// Img Toggle jquery
$("#asl_cheat_button").click(function(){
    if($(this).hasClass("sheet_on")){
        $(this).removeClass("sheet_on")
        $("#asl_cheat_sheet").removeClass("hidden")
        $("#asl_cheat_sheet").addClass("visible")
    }
    else{
        $(this).addClass("sheet_on")
        $("#asl_cheat_sheet").removeClass("visible")
        $("#asl_cheat_sheet").addClass("hidden")
    }
});

