/*

    .----.                    _..._                                                     .-'''-.
   / .--./    .---.        .-'_..._''.                          _______                '   _    \
  ' '         |   |.--.  .' .'      '.\     __.....__           \  ___ `'.           /   /` '.   \_________   _...._
  \ \         |   ||__| / .'            .-''         '.    ,.--. ' |--.\  \         .   |     \  '\        |.'      '-.
   `.`'--.    |   |.--.. '             /     .-''"'-.  `. //    \| |    \  ' .-,.--.|   '      |  '\        .'```'.    '.
     `'-. `.  |   ||  || |            /     /________\   \\\    /| |     |  '|  .-. \    \     / /  \      |       \     \
         `. \ |   ||  || |            |                  | `'--' | |     |  || |  | |`.   ` ..' /    |     |        |    |
           \ '|   ||  |. '            \    .-------------' ,.--. | |     ' .'| |  | |   '-...-'`     |      \      /    .
            | |   ||  | \ '.          .\    '-.____...---.//    \| |___.' /' | |  '-                 |     |\`'-.-'   .'
            | |   ||__|  '. `._____.-'/ `.             .' \\    /_______.'/  | |                     |     | '-....-'`
           / /'---'        `-.______ /    `''-...... -'    `'--'\_______|/   | |                    .'     '.
     /...-'.'                       `                                        |_|                  '-----------'
    /--...-'

    Slice:Drop - Instantly view scientific and medical imaging data in 3D.

     http://slicedrop.com

    Copyright (c) 2012 The Slice:Drop and X Toolkit Developers <dev@goXTK.com>

    Slice:Drop is licensed under the MIT License:
      http://www.opensource.org/licenses/mit-license.php

    CREDITS: http://slicedrop.com/LICENSE

*/

/**
 * Setup all UI elements once the loading was completed.
 */
function setupUi() {

}

function toggleVolumeRendering() {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	volume.volumeRendering = !volume.volumeRendering;
}

function thresholdVolume(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	if (event == null){
		if (ui[0] != null){
			volume.lowerThreshold = ui[0];
			volume.upperThreshold = ui[1];		
		}
		if (ui[1] != null){
			volume.lowerThreshold = ui[0];
			volume.upperThreshold = ui[1];		
		}
	} else {
		volume.lowerThreshold = ui.values[0];
		volume.upperThreshold = ui.values[1];
	}
}

function windowLevelVolume(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	
	volume.windowLow = ui.values[0];
	volume.windowHigh = ui.values[1];
}

function opacity3dVolume(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	if (event == null){
		volume.opacity = ui;	
	} else {
		volume.opacity = ui.value / 100;
	}
}

function toggleAxialSliceVisibility() {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	volume.children[2].children[Math.floor(volume.indexZ)].visible = !volume.children[2].children[Math.floor(volume.indexZ)].visible;
}

function toggleCoronalSliceVisibility() {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	volume.children[1].children[Math.floor(volume.indexY)].visible = !volume.children[1].children[Math.floor(volume.indexY)].visible;
}

function toggleSagittalSliceVisibility() {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	volume.children[0].children[Math.floor(volume.indexX)].visible = !volume.children[0].children[Math.floor(volume.indexX)].visible;
}

function volumeslicingX(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	volume.indexX = Math.floor($('#yellow_slider').slider("option", "value"));
}

function volumeslicingY(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	volume.indexY = Math.floor($('#red_slider').slider("option", "value"));
}

function volumeslicingZ(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	volume.indexZ = Math.floor($('#green_slider').slider("option", "value"));
}

function fgColorVolume(hex, rgb) {
  var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
  if (!volume) {
    return;
  }

  volume.maxColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushVolume.bind(RT, 'maxColor', volume.maxColor), 150);

  }

}

function bgColorVolume(hex, rgb) {
  var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
  if (!volume) {
    return;
  }

  volume.minColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushVolume.bind(RT, 'minColor', volume.minColor), 150);

  }

}

//
// LABELMAP
//
function opacityLabelmap(event, ui) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
	return;
	}
	if (event == null){
		volume.labelmap.opacity = ui;
	} else {
		volume.labelmap.opacity = ui.value / 100;
	}
}

function toggleLabelmapVisibility(label) {
	var volume = _ATLAS_.volumes[_ATLAS_.currentVolume];
	if (!volume) {
		return;
	}
	if (label == "all") {
		label = null;
		volume.labelmap.opacity = 0.5;
	} else if (label == "none") {
		label = null;
		volume.labelmap.opacity = 0;
	} else {
		volume.labelmap.opacity = 0.5;
	}
	volume.labelmap.showOnly = label;

	// remember opacity
	_ATLAS_.labelOpacity = volume.labelmap.opacity;
	
	// update label caption
	var labelname = volume.labelmap.colortable.get(label)[0];
	var _r = parseInt(volume.labelmap.colortable.get(label)[1] * 255, 10);
	var _g = parseInt(volume.labelmap.colortable.get(label)[2] * 255, 10);
	var _b = parseInt(volume.labelmap.colortable.get(label)[3] * 255, 10);
	$('#anatomy_caption').html(labelname);
	$('#anatomy_caption').css('color', 'rgb( ' + _r + ',' + _g + ',' + _b + ' )' );
}

//
// MESH
//
function toggleMeshVisibility(label) {

	if (!_ATLAS_.meshes[_ATLAS_.currentVolume][label]) {
		// load mesh
		var m = new X.mesh();
		m.file = "data/"+_ATLAS_.steps[_ATLAS_.currentVolume]+"/"+label;
		m.caption = label.replace(".vtk","").split("_")[2].split();//.replace("Model_","")
		console.log(label);
		r0.add(m);
		// grab label value
		var labelvalue = label.replace("Model_","").split("_")[0];
		m.color = _ATLAS_.volumes[0].labelmap.colortable.get(labelvalue).slice(0).splice(1,3);
		_ATLAS_.meshes[_ATLAS_.currentVolume][label] = m;
		_ATLAS_.meshes[_ATLAS_.currentVolume][label].visible = false;
		_ATLAS_.meshes[_ATLAS_.currentVolume][label].opacity = _ATLAS_.meshOpacity;
	}
	// show the mesh
	_ATLAS_.meshes[_ATLAS_.currentVolume][label].visible = !_ATLAS_.meshes[_ATLAS_.currentVolume][label].visible;
}

function meshColor(hex, rgb) {

  if (!mesh) {
    return;
  }

  mesh.color = [rgb.r / 255, rgb.g / 255, rgb.b / 255];

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushMesh.bind(RT, 'color', mesh.color), 150);

  }
}

function opacityMesh(event, ui) {

	for (var m in _ATLAS_.meshes[_ATLAS_.currentVolume]) {
		if (_ATLAS_.meshes[_ATLAS_.currentVolume][m] != null) {
			_ATLAS_.meshes[_ATLAS_.currentVolume][m].opacity = ui.value / 100;		
		}
	}
	
	_ATLAS_.meshOpacity = ui.value / 100;

	//_ATLAS_.meshes[_ATLAS_.currentVolume]['Model_3_Left-Cerebral-Cortex.vtk'].opacity = ui.value / 100;
/*  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushMesh.bind(RT, 'opacity', mesh.opacity), 150);

  }
 */
}

function thresholdScalars(event, ui) {

  if (!mesh) {
    return;
  }

  mesh.scalars.lowerThreshold = ui.values[0] / 100;
  mesh.scalars.upperThreshold = ui.values[1] / 100;

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushScalars.bind(RT, 'lowerThreshold', mesh.scalars.lowerThreshold), 150);
    clearTimeout(RT._updater2);
    RT._updater2 = setTimeout(RT.pushScalars.bind(RT, 'upperThreshold', mesh.scalars.upperThreshold), 150);

  }

}

function scalarsMinColor(hex, rgb) {

  if (!mesh) {
    return;
  }

  mesh.scalars.minColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushScalars.bind(RT, 'minColor', mesh.scalars.minColor), 150);

  }

}

function scalarsMaxColor(hex, rgb) {

  if (!mesh) {
    return;
  }

  mesh.scalars.maxColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushScalars.bind(RT, 'maxColor', mesh.scalars.maxColor), 150);

  }

}

//
// Fibers
//
function toggleFibersVisibility() {

  if (!fibers) {
    return;
  }

  fibers.visible = !fibers.visible;

  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushFibers.bind(RT, 'visible', fibers.visible), 150);

  }


}

function thresholdFibers(event, ui) {

  if (!fibers) {
    return;
  }

  fibers.scalars.lowerThreshold = ui.values[0];
  fibers.scalars.upperThreshold = ui.values[1];
  if (RT.linked) {

    clearTimeout(RT._updater);
    RT._updater = setTimeout(RT.pushFibersScalars.bind(RT, 'lowerThreshold', fibers.scalars.lowerThreshold), 150);
    clearTimeout(RT._updater2);
    RT._updater2 = setTimeout(RT.pushFibersScalars.bind(RT, 'upperThreshold', fibers.scalars.upperThreshold), 150);

  }

}



function scene_picking_check() {

  // return the currently picked model
  return $('.scene_picker').filter(':visible').html();

}