package inventory

import (
	"github.com/markus-wa/demoinfocs-golang/common"
)
//TODO: Json marshal/unmarshaling for writing to file


type Equipment struct{
	name string
	count int
}

var equipmentList []Equipment
var equipmentKeys []common.EquipmentElement // To be able to easier insert weapons correctly when retrieving weapons

//TODO: Make sure we have same array structure for image output as for this output,
// in regards to making the array indices the same for every equipment type
// Maps seem to have random order? Maybe need to hardcode this instead
func InitEquipment(){
	for equipment := range common.EquipmentElementNames(){
		newEquipment := Equipment{
			name: equipment.String(),
			count: 0,
		}
		equipmentList = append(equipmentList, newEquipment)
		equipmentKeys = append(equipmentKeys, common.MapEquipment(equipment.String()))
	}
}

func GetEquipmentList() []Equipment{
	return equipmentList
}

func GetEquipmentKeys() []common.EquipmentElement{
	return equipmentKeys
}

