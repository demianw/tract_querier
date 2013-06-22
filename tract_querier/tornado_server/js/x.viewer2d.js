

// create 2D viewer controls

/*$(function() {
	
	// create the sliders for the 2D sliders
	$("#yellow_slider").slider({
		slide: volumeslicingX
	});
	$("#yellow_slider .ui-slider-handle").unbind('keydown');
	
	$("#red_slider").slider({
		slide: volumeslicingY
	});
	$("#red_slider .ui-slider-handle").unbind('keydown');
	
	$("#green_slider").slider({
		slide: volumeslicingZ
	});
	$("#green_slider .ui-slider-handle").unbind('keydown');
	
});*/


// initialize the 2D viewers
function init_viewer2d() {

	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];

	// X Slice
	if (sliceX) {
		sliceX.destroy();
	}
	sliceX = new X.renderer2D();
	sliceX.container = 'sliceX';
	sliceX.orientation = 'X';
	sliceX.init();
	sliceX.add(volume);
	sliceX.render();
	sliceX.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceX), 100);
	};

	// Y Slice
	if (sliceY) {
		sliceY.destroy();
	}
	sliceY = new X.renderer2D();
	sliceY.container = 'sliceY';
	sliceY.orientation = 'Y';
	sliceY.init();
	sliceY.add(volume);
	sliceY.render();
	sliceY.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceY), 100);
	};

	// Z Slice
	if (sliceZ) {
		sliceZ.destroy();
	}
	sliceZ = new X.renderer2D();
	sliceZ.container = 'sliceZ';
	sliceZ.orientation = 'Z';
	sliceZ.init();
	sliceZ.add(volume);
	sliceZ.render();
	sliceZ.interactor.onMouseMove = function() {
		if (_ATLAS_.hover){
			clearTimeout(_ATLAS_.hover);
		}
		_ATLAS_.hover = setTimeout(on2DHover.bind(this, sliceZ), 100);
	  };

	// update 2d slice sliders
	var dim = volume.dimensions;
	$("#yellow_slider").slider("option", "disabled", false);
	$("#yellow_slider").slider("option", "min", 0);
	$("#yellow_slider").slider("option", "max", dim[0] - 1);
	$("#yellow_slider").slider("option", "value", volume.indexX);
	$("#red_slider").slider("option", "disabled", false);
	$("#red_slider").slider("option", "min", 0);
	$("#red_slider").slider("option", "max", dim[1] - 1);
	$("#red_slider").slider("option", "value", volume.indexY);
	$("#green_slider").slider("option", "disabled", false);
	$("#green_slider").slider("option", "min", 0);
	$("#green_slider").slider("option", "max", dim[2] - 1);
	$("#green_slider").slider("option", "value", volume.indexZ);

}//init_viewer2d()


// show labels on hover
function on2DHover(renderer) {

	// check that feature is enabled
	if (!_ATLAS_.hoverLabelSelect){
		return;
	}

	// get cursor position
	var mousepos = renderer.interactor.mousePosition;
	var ijk = renderer.xy2ijk(mousepos[0], mousepos[1]);
	if (!ijk) {
		return;
	}

	//
	var orientedIJK = ijk.slice();
	orientedIJK[0] = ijk[2];
	orientedIJK[1] = ijk[1];
	orientedIJK[2] = ijk[0];

	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	
	// get the number associated with the label
	var labelvalue = volume.labelmap.image[orientedIJK[0]][orientedIJK[1]][orientedIJK[2]];
	if (!labelvalue || labelvalue == 0) {
		return;
	}
	//console.log(labelvalue);

	
	volume.labelmap.opacity = 0.6;
	volume.labelmap.showOnly = labelvalue;
	var labelname = volume.labelmap.colortable.get(labelvalue)[0];
	var _r = parseInt(volume.labelmap.colortable.get(labelvalue)[1] * 255, 10);
	var _g = parseInt(volume.labelmap.colortable.get(labelvalue)[2] * 255, 10);
	var _b = parseInt(volume.labelmap.colortable.get(labelvalue)[3] * 255, 10);
	$('#anatomy_caption').html(labelname);
	$('#anatomy_caption').css('color', 'rgb( ' + _r + ',' + _g + ',' + _b + ' )' );
}
