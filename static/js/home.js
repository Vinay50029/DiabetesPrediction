function setActionType(actionType) {
    const form = document.getElementById("userForm");
    document.getElementById("action_type").value = actionType;
    console.log("Action Type Set To:", actionType);
    if (form.checkValidity()) {
      form.submit();
    } else {
      form.reportValidity();
    }
  }


